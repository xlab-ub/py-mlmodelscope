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
from mlharness_tools.accuracy_imagenet import calculate_accuracy
from mlharness_tools.process_benchmark_results import process_benchmark_results
from mlmodelscope.dataloader import DataLoader 
from mlmodelscope.outputprocessor import OutputProcessor 
from mlmodelscope.processor_name import get_cpu_name, get_gpu_name 
import pydldataset 
from tracer import Tracer 

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


NANO_SEC = 1e9
MILLI_SEC = 1000

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
last_timeing = []

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}


TRACE_LEVEL = ( "NO_TRACE",
                "APPLICATION_TRACE",
                "MODEL_TRACE",          # pipelines within model
                "FRAMEWORK_TRACE",      # layers within framework
                "ML_LIBRARY_TRACE",     # cudnn, ...
                "SYSTEM_LIBRARY_TRACE", # cupti
                "HARDWARE_TRACE",       # perf, papi, ...
                "FULL_TRACE")           # includes all of the above)
BACKENDS = ("pytorch", "onnxruntime", "tensorflow", "mxnet", "jax") 

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='cnn', choices=['coco', 'imagenet', 'squad', 'brats2019', 'cnn'], help="select accuracy script for dataset")
    parser.add_argument("--scenario", default="SingleStream",
                        help="mlcommons inference benchmark scenario, one of " + str(list(SCENARIO_MAP.keys())))
    # in MLPerf the default max-batchsize value is 128, but in Onnxruntime some models can only support size of 1
    parser.add_argument("--max_batchsize", type=int, default=1, help="max batch size in a single inference")
    parser.add_argument("--backend", default='pytorch', choices=BACKENDS, help="runtime to use")
    parser.add_argument("--task", type=str, nargs='?', default="summarization", help="The name of the task to predict.") 
    parser.add_argument("--model_names", type=str, nargs='+', default=["mlperf_resnet_50"], help="all the models you want to compare") 
    parser.add_argument("--mlperf_model_names", type=str, nargs='+', default=None, help="all the models you want to compare") 
    parser.add_argument("--qps", type=int, help="target qps")
    # parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--accuracy", type=bool, default=False, help="enable accuracy pass")
    parser.add_argument("--find_peak_performance", action="store_true", help="enable finding peak performance pass")

    # file to use mlperf rules compliant parameters
    parser.add_argument("--mlperf_conf", default="./inference/mlperf.conf", help="mlperf rules config")
    # file for user LoadGen settings such as target QPS
    # parser.add_argument("--user_conf", default="./inference/vision/classification_and_detection/user.conf", help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--user_conf", default="./inference/language/gpt-j/user.conf", help="user config for user LoadGen settings such as target QPS")
    # log path for loadgen
    parser.add_argument("--log_dir", default='./logs')
    
    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument("--max_latency", type=float, help="mlperf max latency in pct tile")
    parser.add_argument("--samples_per_query", type=int, help="mlperf multi-stream sample per query")


    # MLHarness Parameters
    parser.add_argument("--use_gpu", type=int, default=1, help="enable gpu for inference")
    parser.add_argument("--gpu_id", type=int, default=0, help="which GPU")
    parser.add_argument("--trace_level", type=str, nargs='?', default="NO_TRACE", choices=TRACE_LEVEL, help="MLModelScope Trace Level") 
    parser.add_argument("--gpu_trace", type=str, nargs='?', default="false", choices=["false", "true"], help="Whether to trace GPU activities") 
    parser.add_argument("--save_trace_result", type=str, nargs='?', default="false", choices=["false", "true"], help="Whether to save the trace result")
    parser.add_argument("--save_trace_result_path", type=str, nargs='?', default="trace_result.txt", help="The path of the trace result file")
    # py-mlmodelscope Parameters
    parser.add_argument("--security_check", type=str, nargs='?', default="false", choices=["false", "true"], help="Whether to perform security check on the model file")
    parser.add_argument("--config_file", type=str, nargs='?', default="false", choices=["false", "true"], help="Whether to use config file (.json)") 
    parser.add_argument("--config_file_path", type=str, nargs='?', default="config.json", help="The path of the config file") 
    # Modality Specific
    # inv_map for object detection
    parser.add_argument("--use_inv_map", action="store_true", help="use inv_map for object detection")

    args = parser.parse_args()

    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scenarios:" + str(list(SCENARIO_MAP.keys())))

    return args

def add_results(final_results, name, result_dict, result_list, took, show_accuracy=False, show_results=True):
    percentiles = [50., 80., 90., 95., 99., 99.9]
    buckets = np.percentile(result_list, percentiles).tolist()
    buckets_str = ",".join(["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)])

    if result_dict["total"] == 0:
        result_dict["total"] = len(result_list)

    # this is what we record for each run
    result = {
        # "took": took,
        "mean": np.mean(result_list),
        "percentiles": {str(k): v for k, v in zip(percentiles, buckets)},
        "qps": len(result_list) / took,
        "count": len(result_list),
        "good_items": result_dict["good"],
        "total_items": result_dict["total"],
    }
    acc_str = ""
    if show_accuracy:
        result["accuracy"] = 100. * result_dict["good"] / result_dict["total"]
        acc_str = ", acc={:.3f}%".format(result["accuracy"])
        if "mAP" in result_dict:
            result["mAP"] = 100. * result_dict["mAP"]
            acc_str += ", mAP={:.3f}%".format(result["mAP"])

    # add the result to the result dict
    final_results[name] = result

    # to stdout
    if show_results:
        print("{} qps={:.2f}, mean={:.4f}, time={:.3f}{}, queries={}, tiles={}".format(
            name, result["qps"], result["mean"], took, acc_str,
            len(result_list), buckets_str))
    


def run_harness(args, benchmark_model, mlperf_model_name=None):
    last_loaded = -1
    global last_timeing

    result_timeing = []
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
    model_name = benchmark_model
    config = None 
    if args.config_file == "true":
        config_file_path = args.config_file_path 
        try: 
            with open(config_file_path, 'r') as f:
                config = json.load(f)
                print(f"config file {config_file_path} is loaded")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"config file {config_file_path} is not loaded: {e}") 
    architecture = 'cpu' if args.use_gpu == 0 else 'gpu' 
    trace_level = args.trace_level
    gpu_trace = True if (TRACE_LEVEL.index(trace_level) >= TRACE_LEVEL.index("SYSTEM_LIBRARY_TRACE")) and (args.gpu_trace == "true") else False 
    security_check = True if args.security_check == "true" else False 

    save_trace_result = True if (args.save_trace_result == "true") and (trace_level != "NO_TRACE") else False 
    save_trace_result_path = args.save_trace_result_path if save_trace_result else None

    tracer, root_span, ctx = Tracer.create(trace_level=trace_level, save_trace_result_path=save_trace_result_path)
    root_span.set_attribute("cpu_name", get_cpu_name()) 
    
    c = None 
    if architecture == "gpu" and gpu_trace and tracer.is_trace_enabled("SYSTEM_LIBRARY_TRACE"): 
        root_span.set_attribute("gpu_name", get_gpu_name()) 
        from pycupti import CUPTI 
        c = CUPTI(tracer=tracer) 
        print("CUPTI version", c.cuptiGetVersion())

    output_processor = OutputProcessor() 

    #load backend

    user = 'default' 
    if backend == 'pytorch': 
        from mlmodelscope.pytorch_agent import PyTorch_Agent 
        agent = PyTorch_Agent(task, model_name, architecture, tracer, ctx, security_check, config, user, c) 
    elif backend == 'tensorflow': 
        from mlmodelscope.tensorflow_agent import TensorFlow_Agent 
        agent = TensorFlow_Agent(task, model_name, architecture, tracer, ctx, security_check, config, user) 
    elif backend == 'onnxruntime': 
        from mlmodelscope.onnxruntime_agent import ONNXRuntime_Agent 
        agent = ONNXRuntime_Agent(task, model_name, architecture, tracer, ctx, security_check, config, user) 
    elif backend == 'mxnet': 
        from mlmodelscope.mxnet_agent import MXNet_Agent 
        agent = MXNet_Agent(task, model_name, architecture, tracer, ctx, security_check, config, user) 
    elif backend == 'jax':
        from mlmodelscope.jax_agent import JAX_Agent
        agent = JAX_Agent(task, model_name, architecture, tracer, ctx, security_check, config, user)
    else: 
      raise NotImplementedError(f"{backend} agent is not supported") 
    
    final_results = {
        "runtime": backend,
        "time": int(time.time()),
        "args": vars(args),
        "cmdline": str(args),
    }

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
    print(f"count: {count}")

    # warmup
    dataset.load([0]) 
    for _ in range(5):
        img = dataset.get_samples([0])
        # _ = backend.predict({backend.inputs[0]: img})
        agent.predict(0, DataLoader(img, args.max_batchsize), output_processor) 
    dataset.unload(None)

    scenario = SCENARIO_MAP[args.scenario]

    # for lg.ConstructSUT() 
    def issue_queries(query_samples):

        idx = np.array([q.index for q in query_samples]).astype(np.int32)
        query_id = [q.id for q in query_samples]
        if args.dataset == 'brats2019':
            start = time.time()
            response_array_refs = []
            response = []
            for index, qid in enumerate(query_id):
                # processed_results = so.IssueQuery(1, idx[i][np.newaxis])
                processed_results = agent.predict(0, DataLoader(dataset.get_samples(idx[index][np.newaxis]), args.max_batchsize), output_processor, mlharness=True) 
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
            processed_results = agent.predict(0, DataLoader(dataset.get_samples(idx), args.max_batchsize), output_processor, mlharness=True) 
            result_timeing.append(time.time() - start)
            # processed_results = json.loads(processed_results.decode('utf-8'))
            response_array_refs = []
            response = []
            for index, qid in enumerate(query_id):
                if args.dataset == 'coco': 
                    for j in range(len(processed_results[index])): 
                        processed_results[index][j] = [idx[index]] + processed_results[index][j] 
                dtype = np.int64 if args.dataset == 'cnn' else np.float32 
                response_array = array.array("B", np.array(processed_results[index], dtype).tobytes()) 
                response_array_refs.append(response_array)
                bi = response_array.buffer_info()
                response.append(lg.QuerySampleResponse(qid, bi[0], bi[1]))

            lg.QuerySamplesComplete(response)
    
    # for lg.ConstructSUT() 
    def flush_queries():
        pass
    
    settings = lg.TestSettings()

    if mlperf_model_name:
        settings.FromConfig(mlperf_conf, mlperf_model_name, args.scenario)
        settings.FromConfig(user_conf, mlperf_model_name, args.scenario)


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
    log_settings.enable_trace = True
    # lg.StartTest(sut, qsl, settings)
    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)

    if not last_timeing:
        last_timeing = result_timeing

    if args.accuracy:
        accuracy_script_paths = {'coco': os.path.realpath('./inference/vision/classification_and_detection/tools/accuracy-coco.py'),
                        'imagenet': os.path.realpath('./inference/vision/classification_and_detection/tools/accuracy-imagenet.py'),
                        'squad': os.path.realpath('./inference/language/bert/accuracy-squad.py'),
                        'brats2019': os.path.realpath('./inference/vision/medical_imaging/3d-unet/accuracy-brats.py'),
                        'cnn': os.path.realpath('./inference/language/gpt-j/evaluation.py')} 
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
        elif args.dataset == 'cnn':   # cnn
            subprocess.check_call('python3 {} --mlperf-accuracy-file {} --dataset-file {}'.format(accuracy_script_path, accuracy_file_path, os.path.join(data_dir, 'cnn_eval.json')), shell=True)
        else:
            raise RuntimeError('Dataset not Implemented.')

    result_dict = calculate_accuracy(f"{log_dir}/mlperf_log_accuracy.json",f"{data_dir}/val_map.txt", scenario=args.scenario)
    
    print(f"last_timeing len: {len(last_timeing)}")
    add_results(final_results, "{}".format(scenario) , result_dict, last_timeing, time.time() - dataset.last_loaded, args.accuracy )

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)

    return final_results


def main():

    args = get_args()
    log.info(args)

    benchmark_results = []

    if args.mlperf_model_names:
        for model_name, mlperf_model_name in zip(args.model_names, args.mlperf_model_names):
            if mlperf_model_name == "None":
                mlperf_model_name = None
            summary_result = run_harness(args, model_name, mlperf_model_name)
            benchmark_results.append(summary_result)
    
    process_benchmark_results(benchmark_results)

if __name__ == "__main__":
    main()
