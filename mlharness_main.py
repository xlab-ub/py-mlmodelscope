"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import array
import collections
import json
import logging
import os
import sys
import threading
import time
from queue import Queue
import subprocess
import mlperf_loadgen as lg
import numpy as np

from opentelemetry import trace 
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator 
from opentelemetry.sdk.resources import SERVICE_NAME, Resource 
from opentelemetry.sdk.trace import TracerProvider 
from opentelemetry.sdk.trace.export import BatchSpanProcessor 
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter 

from mlmodelscope.dataloader import DataLoader 
import pydldataset 

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

NANO_SEC = 1e9
MILLI_SEC = 1000

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

last_timeing = []
result_timeing = []
last_loaded = -1

TRACE_LEVEL = ( "NO_TRACE",
                "APPLICATION_TRACE",
                "MODEL_TRACE",          # pipelines within model
                "FRAMEWORK_TRACE",      # layers within framework
                "ML_LIBRARY_TRACE",     # cudnn, ...
                "SYSTEM_LIBRARY_TRACE", # cupti
                "HARDWARE_TRACE",       # perf, papi, ...
                "FULL_TRACE")           # includes all of the above)
BACKENDS = ("pytorch", "onnxruntime", "tensorflow", "mxnet")

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='coco', choices=['coco', 'imagenet', 'squad', 'brats2019'], help="select accuracy script for dataset")
    parser.add_argument("--scenario", default="SingleStream",
                        help="mlcommons inference benchmark scenario, one of " + str(list(SCENARIO_MAP.keys())))
    # in MLPerf the default max-batchsize value is 128, but in Onnxruntime some models can only support size of 1
    parser.add_argument("--max_batchsize", type=int, default=1, help="max batch size in a single inference")
    parser.add_argument("--backend", default='tensorflow', choices=BACKENDS, help="runtime to use")
    parser.add_argument("--task", type=str, nargs='?', default="image_object_detection", help="The name of the task to predict.") 
    parser.add_argument("--model_name", type=str, nargs='?', default="ssdlite_mobilenet_v2_coco", help="The name of the model") 
    parser.add_argument("--qps", type=int, help="target qps")
    # parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--accuracy", default=True, help="enable accuracy pass")
    parser.add_argument("--find_peak_performance", action="store_true", help="enable finding peak performance pass")

    # file to use mlperf rules compliant parameters
    parser.add_argument("--mlperf_conf", default="./inference/mlperf.conf", help="mlperf rules config")
    # file for user LoadGen settings such as target QPS
    parser.add_argument("--user_conf", default="./inference/vision/classification_and_detection/user.conf", help="user config for user LoadGen settings such as target QPS")
    # log path for loadgen
    parser.add_argument("--log_dir", default='./logs')
    
    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, default=10, help="dataset items to use")
    parser.add_argument("--max_latency", type=float, help="mlperf max latency in pct tile")
    parser.add_argument("--samples_per_query", type=int, help="mlperf multi-stream sample per query")


    # MLHarness Parameters
    parser.add_argument("--use_gpu", type=int, default=0, help="enable gpu for inference")
    parser.add_argument("--gpu_id", type=int, default=0, help="which GPU")
    parser.add_argument("--trace_level", choices=TRACE_LEVEL, default="NO_TRACE", help="MLModelScope Trace Level")
    # Modality Specific
    # inv_map for object detection
    parser.add_argument("--use_inv_map", action="store_true", help="use inv_map for object detection")

    args = parser.parse_args()

    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scenarios:" + str(list(SCENARIO_MAP.keys())))

    return args

def main():
    resource = Resource(attributes={
        SERVICE_NAME: "mlharness"
    }) 
    trace.set_tracer_provider(TracerProvider(resource=resource)) 
    # https://opentelemetry-python.readthedocs.io/en/latest/exporter/otlp/otlp.html 
    span_processor = BatchSpanProcessor(OTLPSpanExporter(endpoint='http://localhost:4317', insecure=True)) 
    trace.get_tracer_provider().add_span_processor(span_processor) 

    tracer = trace.get_tracer(__name__) 
    prop = TraceContextTextMapPropagator() 
    carrier = {} 

    global last_timeing
    global last_loaded
    global result_timeing

    args = get_args()

    log.info(args)

    # --count applies to accuracy mode only and can be used to limit the number of images
    # for testing. For perf model we always limit count to 200.
    count_override = False
    count = args.count
    if count:
        count_override = True

    # dataset to use 
    dataset_name = args.dataset
    dataset = pydldataset.load(dataset_name, count=count) 

    # load model 
    backend = args.backend 
    task = args.task 
    model_name = args.model_name 
    architecture = 'cpu' if args.use_gpu == 0 else 'gpu' 

    if backend == 'pytorch': 
      from mlmodelscope.pytorch_agent import PyTorch_Agent 
      agent = PyTorch_Agent(task, model_name, architecture, tracer, prop, carrier) 
    elif backend == 'tensorflow': 
      from mlmodelscope.tensorflow_agent import TensorFlow_Agent 
      agent = TensorFlow_Agent(task, model_name, architecture, tracer, prop, carrier) 
    elif backend == 'onnxruntime': 
      from mlmodelscope.onnxruntime_agent import ONNXRuntime_Agent 
      agent = ONNXRuntime_Agent(task, model_name, architecture, tracer, prop, carrier) 
    elif backend == 'mxnet': 
      from mlmodelscope.mxnet_agent import MXNet_Agent 
      agent = MXNet_Agent(task, model_name, architecture, tracer, prop, carrier) 
    else: 
      raise NotImplementedError(f"{backend} agent is not supported") 

    mlperf_conf = os.path.abspath(args.mlperf_conf)
    if not os.path.exists(mlperf_conf):
        log.error("{} not found".format(mlperf_conf))
        sys.exit(1)

    user_conf = os.path.abspath(args.user_conf)
    if not os.path.exists(user_conf):
        log.error("{} not found".format(user_conf))
        sys.exit(1)

    log_dir = None

    if args.log_dir:
        log_dir = os.path.abspath(args.log_dir)
        os.makedirs(log_dir, exist_ok=True)

    #
    # make one pass over the dataset to validate accuracy
    #
    count = dataset.get_item_count() 

    # warmup
    dataset.load([0]) 
    for _ in range(5):
        img = dataset.get_samples([0])
        # _ = backend.predict({backend.inputs[0]: img})
        agent.predict(0, DataLoader(img, args.max_batchsize))
    dataset.unload(None)

    scenario = SCENARIO_MAP[args.scenario]

    # for lg.ConstructSUT() 
    def issue_queries(query_samples):
        global so
        global last_timeing
        global result_timeing
        idx = np.array([q.index for q in query_samples]).astype(np.int32)
        query_id = [q.id for q in query_samples]
        if args.dataset == 'brats2019':
            start = time.time()
            response_array_refs = []
            response = []
            for i, qid in enumerate(query_id):
                # processed_results = so.IssueQuery(1, idx[i][np.newaxis])
                processed_results = agent.predict(0, DataLoader(dataset.get_samples(idx[i][np.newaxis]), args.max_batchsize), mlharness=True) 
                # processed_results = json.loads(processed_results.decode('utf-8'))
                for j in range(len(processed_results[index])): 
                    processed_results[index][j] = [idx[index]] + processed_results[index][j] 
                response_array = array.array("B", np.array(idx[index] + processed_results[0], np.float16).tobytes())
                response_array_refs.append(response_array)
                bi = response_array.buffer_info()
                response.append(lg.QuerySampleResponse(qid, bi[0], bi[1]))
            result_timeing.append(time.time() - start)
            lg.QuerySamplesComplete(response)
        else:
            start = time.time()
            # processed_results = so.IssueQuery(len(idx), idx)
            processed_results = agent.predict(0, DataLoader(dataset.get_samples(idx), args.max_batchsize), mlharness=True)
            result_timeing.append(time.time() - start)
            # processed_results = json.loads(processed_results.decode('utf-8'))
            response_array_refs = []
            response = []
            for index, qid in enumerate(query_id):
                for j in range(len(processed_results[index])): 
                    processed_results[index][j] = [idx[index]] + processed_results[index][j] 
                response_array = array.array("B", np.array(processed_results[index], np.float32).tobytes())
                response_array_refs.append(response_array)
                bi = response_array.buffer_info()
                response.append(lg.QuerySampleResponse(qid, bi[0], bi[1]))

            lg.QuerySamplesComplete(response)
    
    # for lg.ConstructSUT() 
    def flush_queries():
        pass
    
    settings = lg.TestSettings()
    if args.model_name != "":
        settings.FromConfig(mlperf_conf, args.model_name, args.scenario)
        settings.FromConfig(user_conf, args.model_name, args.scenario)
    settings.scenario = scenario
    settings.mode = lg.TestMode.PerformanceOnly
    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    if args.find_peak_performance:
        settings.mode = lg.TestMode.FindPeakPerformance

    if args.time:
        # override the time we want to run
        settings.min_duration_ms = args.time * MILLI_SEC
        settings.max_duration_ms = args.time * MILLI_SEC

    if args.qps:
        qps = float(args.qps)
        settings.server_target_qps = qps
        settings.offline_expected_qps = qps

    if count_override:
        settings.min_query_count = count
        settings.max_query_count = count

    if args.samples_per_query:
        settings.multi_stream_samples_per_query = args.samples_per_query
    if args.max_latency:
        settings.server_target_latency_ns = int(args.max_latency * NANO_SEC)
        settings.multi_stream_target_latency_ns = int(args.max_latency * NANO_SEC)

    # sut = lg.ConstructSUT(issue_queries, flush_queries, process_latencies)
    sut = lg.ConstructSUT(issue_queries, flush_queries)
    qsl = lg.ConstructQSL(count, min(count, 500), dataset.load, dataset.unload) 

    log.info("starting {}".format(scenario))

    log_path = os.path.realpath(args.log_dir)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    # log_settings.enable_trace = True
    # lg.StartTest(sut, qsl, settings)
    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)

    if not last_timeing:
        last_timeing = result_timeing

    if args.accuracy:
        accuracy_script_paths = {'coco': os.path.realpath('./inference/vision/classification_and_detection/tools/accuracy-coco.py'),
                        'imagenet': os.path.realpath('./inference/vision/classification_and_detection/tools/accuracy-imagenet.py'),
                        'squad': os.path.realpath('./inference/language/bert/accuracy-squad.py'),
                        'brats2019': os.path.realpath('./inference/vision/medical_imaging/3d-unet/accuracy-brats.py'),}
        accuracy_script_path = accuracy_script_paths[args.dataset]
        accuracy_file_path = os.path.join(log_dir, 'mlperf_log_accuracy.json')
        data_dir = os.environ['DATA_DIR']
        if args.dataset == 'coco':
            if args.use_inv_map:
                subprocess.check_call('python3 {} --mlperf-accuracy-file {} --coco-dir {} --use-inv-map'.format(accuracy_script_path, accuracy_file_path, data_dir), shell=True)
            else:
                subprocess.check_call('python3 {} --mlperf-accuracy-file {} --coco-dir {}'.format(accuracy_script_path, accuracy_file_path, data_dir), shell=True)
        elif args.dataset == 'imagenet':   # imagenet
            subprocess.check_call('python3 {} --mlperf-accuracy-file {} --imagenet-val-file {}'.format(accuracy_script_path, accuracy_file_path, os.path.join(data_dir, 'val_map.txt')), shell=True)
        elif args.dataset == 'squad':   # squad
            vocab_path = os.path.join(data_dir, 'vocab.txt')
            val_path = os.path.join(data_dir, 'dev-v1.1.json')
            out_path = os.path.join(log_dir, 'predictions.json')
            cache_path = os.path.join(data_dir, 'eval_features.pickle')
            subprocess.check_call('python3 {} --vocab_file {} --val_data {} --log_file {} --out_file {} --features_cache_file {} --max_examples {}'.
            format(accuracy_script_path, vocab_path, val_path, accuracy_file_path, out_path, cache_path, count), shell=True)
        elif args.dataset == 'brats2019':   # brats2019
            base_dir = os.path.realpath('./inference/vision/medical_imaging/3d-unet/build')
            post_dir = os.path.join(base_dir, 'postprocessed_data')
            label_dir = os.path.join(base_dir, 'raw_data/nnUNet_raw_data/Task043_BraTS2019/labelsTr')
            os.makedirs(post_dir, exist_ok=True)
            subprocess.check_call('python3 {} --log_file {} --preprocessed_data_dir {} --postprocessed_data_dir {} --label_data_dir {}'.
            format(accuracy_script_path, accuracy_file_path, data_dir, post_dir, label_dir), shell=True)
        else:
            raise RuntimeError('Dataset not Implemented.')

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)

if __name__ == "__main__":
    main()
