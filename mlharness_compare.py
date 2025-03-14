"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import array
import collections
import torch
import json
import logging
import os
import sys
import threading
import time
from queue import Queue
import subprocess
import requests
import mlperf_loadgen as lg
import numpy as np
from opentelemetry import trace
import shutil
from mlharness_tools.accuracy_imagenet import calculate_accuracy
from mlharness_tools.analyze_trace_result_c import analyze_trace_results
from mlharness_tools.process_trace_results import compare_analysis_results, display_comparison_results, compare_loadgen_results
from mlharness_tools.visualize_results import visualize_latency_model_comparison
from mlmodelscope.dataloader import DataLoader
from mlmodelscope.outputprocessor import OutputProcessor
from mlmodelscope.processor_name import get_cpu_name, get_gpu_name
import pydldataset
from tracer import Tracer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


NANO_SEC = 1e9
MILLI_SEC = 1000

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
last_timeing = []

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

SAMPLER_TYPE = ( "PARENT_BASED",     # Uses ParentBased with TraceIdRatioBased
                "TRACE_ID_RATIO",   # Uses only TraceIdRatioBased 
                "ALWAYS_ON",        # defualt sampler used by opentelemetry
                "ALWAYS_OFF")       # drops all traces 

TRACE_LEVEL = (
    "NO_TRACE",
    "APPLICATION_TRACE",
    "MODEL_TRACE",  # pipelines within model
    "FRAMEWORK_TRACE",  # layers within framework
    "ML_LIBRARY_TRACE",  # cudnn, ...
    "SYSTEM_LIBRARY_TRACE",  # cupti
    "HARDWARE_TRACE",  # perf, papi, ...
    "FULL_TRACE",
)  # includes all of the above)
BACKENDS = ("pytorch", "onnxruntime", "tensorflow", "mxnet", "jax")


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="cnn",
        choices=["coco", "imagenet", "squad", "brats2019", "cnn"],
        help="select accuracy script for dataset",
    )
    parser.add_argument(
        "--scenario",
        default=["SingleStream"],
        nargs="+",
        help="mlcommons inference benchmark scenarios, one or more of "
        + str(list(SCENARIO_MAP.keys())),
    )
    parser.add_argument(
        "--sampler",
        default="ALWAYS_ON",
        nargs="?",
        help="sampler type for opentelemetry",
        choices=SAMPLER_TYPE,
    )
    parser.add_argument(
        "--sampler_ratio",
        type=float,
        default=1.0,
        help="sampling ratio for the sampler",
    )
    # in MLPerf the default max-batchsize value is 128, but in Onnxruntime some models can only support size of 1
    parser.add_argument(
        "--max_batchsize",
        type=int,
        default=1,
        help="max batch size in a single inference",
    )
    parser.add_argument(
        "--backend", default="pytorch", choices=BACKENDS, help="runtime to use"
    )
    parser.add_argument(
        "--task",
        type=str,
        nargs="?",
        default="summarization",
        help="The name of the task to predict.",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=["mlperf_resnet_50"],
        help="all the models you want to compare",
    )
    parser.add_argument(
        "--mlperf_model_names",
        type=str,
        nargs="+",
        default=None,
        help="all the models you want to compare",
    )
    parser.add_argument("--qps", type=int, help="target qps")
    # parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument(
        "--accuracy", type=bool, default=False, help="enable accuracy pass"
    )
    parser.add_argument(
        "--find_peak_performance",
        action="store_true",
        help="enable finding peak performance pass",
    )

    # file to use mlperf rules compliant parameters
    parser.add_argument(
        "--mlperf_conf", default="./inference/mlperf.conf", help="mlperf rules config"
    )
    # file for user LoadGen settings such as target QPS
    # parser.add_argument("--user_conf", default="./inference/vision/classification_and_detection/user.conf", help="user config for user LoadGen settings such as target QPS")
    parser.add_argument(
        "--user_conf",
        default="./inference/language/gpt-j/user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    # log path for loadgen
    parser.add_argument("--log_dir", default="./logs")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument(
        "--max_latency", type=float, help="mlperf max latency in pct tile"
    )
    parser.add_argument(
        "--samples_per_query", type=int, help="mlperf multi-stream sample per query"
    )

    # MLHarness Parameters
    parser.add_argument(
        "--use_gpu", type=int, default=1, help="enable gpu for inference"
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="which GPU")
    parser.add_argument(
        "--trace_level",
        type=str,
        nargs="?",
        default="NO_TRACE",
        choices=TRACE_LEVEL,
        help="MLModelScope Trace Level",
    )
    parser.add_argument(
        "--compare_trace",
        type=str,
        default="true",
        choices=["false", "true"],
        
    )
    parser.add_argument(
        "--save_plots",
        type=str,
        default="true",
        choices=["false", "true"],
        
    )
    parser.add_argument(
        "--include_warmup_trace",
        type=str,
        nargs="?",
        default="false",
        choices=["false", "true"],
        help="Whether to include the warmup traces",
    )
    parser.add_argument(
        "--gpu_trace",
        type=str,
        nargs="?",
        default="false",
        choices=["false", "true"],
        help="Whether to trace GPU activities",
    )
    parser.add_argument(
        "--save_trace_result",
        type=str,
        nargs="?",
        default="false",
        choices=["false", "true"],
        help="Whether to save the trace result",
    )
    parser.add_argument(
        "--save_trace_result_path",
        type=str,
        nargs="?",
        default="trace_result.txt",
        help="The path of the trace result file",
    )

    # kernal params for model comparison
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top items to include in comparisons",
    )
    parser.add_argument(
        "--top_k_avg_time_consuming_kernels",
        default="true",
        choices=["false", "true"],
        help="Whether to include top_k_avg_time_consuming_kernels in model comparision",
    )
    parser.add_argument(
        "--top_k_total_time_consuming_kernels",
        default="true",
        choices=["false", "true"],
        help="Whether to include top_k_total_time_consuming_kernels in model comparison",
    )
    parser.add_argument(
        "--top_k_total_time_consuming_kernels_with_count",
        default="true",
        choices=["false", "true"],
        help="Whether to include top_k_total_time_consuming_kernels_with_count in model comparison",
    )
    parser.add_argument(
        "--average_time_for_kernel",
        default="true",
        choices=["false", "true"],
        help="Whether to include average_time_for_kernel in model comparison",
    )
    parser.add_argument(
        "--time_consuming_layers_by_depth",
        default="true",
        choices=["false", "true"],
        help="Whether to include time_consuming_layers_by_depth in model comparison",
    )
    parser.add_argument(
        "--average_time_for_layer",
        default="true",
        choices=["false", "true"],
        help="Whether to include average_time_for_layer in model comparison",
    )
    parser.add_argument(
        "--find_kernels_to_optimize",
        default="true",
        choices=["false", "true"],
        help="Whether to find kernels to optimize in model comparison",
    )
    parser.add_argument(
        "--layerwise_kernel",
        default="true",
        choices=["false", "true"],
        help="Whether to include layerwise_kernel in model comparison",
    )
    # py-mlmodelscope Parameters
    parser.add_argument(
        "--security_check",
        type=str,
        nargs="?",
        default="false",
        choices=["false", "true"],
        help="Whether to perform security check on the model file",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        nargs="?",
        default="false",
        choices=["false", "true"],
        help="Whether to use config file (.json)",
    )
    parser.add_argument(
        "--config_file_path",
        type=str,
        nargs="?",
        default="config.json",
        help="The path of the config file",
    )
    # Modality Specific
    # inv_map for object detection
    parser.add_argument(
        "--use_inv_map", action="store_true", help="use inv_map for object detection"
    )

    args = parser.parse_args()

    for scenario in args.scenario:
        if scenario not in SCENARIO_MAP:
            parser.error(f"Invalid scenario: {scenario}. Valid scenarios: {list(SCENARIO_MAP.keys())}")

    return args


def add_results(
    final_results,
    name,
    result_dict,
    result_list,
    took,
    show_accuracy=False,
    show_results=True,
):
    percentiles = [50.0, 80.0, 90.0, 95.0, 99.0, 99.9]
    buckets = np.percentile(result_list, percentiles).tolist()
    buckets_str = ",".join(
        ["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)]
    )

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
        result["accuracy"] = 100.0 * result_dict["good"] / result_dict["total"]
        acc_str = ", acc={:.3f}%".format(result["accuracy"])
        if "mAP" in result_dict:
            result["mAP"] = 100.0 * result_dict["mAP"]
            acc_str += ", mAP={:.3f}%".format(result["mAP"])

    # add the result to the result dict
    final_results[name] = result

    # to stdout
    if show_results:
        print(
            "{} qps={:.2f}, mean={:.4f}, time={:.3f}{}, queries={}, tiles={}".format(
                name,
                result["qps"],
                result["mean"],
                took,
                acc_str,
                len(result_list),
                buckets_str,
            )
        )


def run_harness(args, benchmark_model, arg_scenario, mlperf_model_name=None):
    last_loaded = -1
    global last_timeing

    result_timeing = []
    # --count applies to accuracy mode only and can be used to limit the number of images
    # for testing. For perf model we always limit count to 200.
    count_override = False
    include_warmup_trace = args.include_warmup_trace == "true"
    print(include_warmup_trace, "include warmup trace")

    count = args.count
    if count:
        count_override = True
    # dataset to use
    dataset_name = args.dataset
    try:
        dataset = pydldataset.load(dataset_name, count=count)
    except Exception as e:
        print("Dataset failed to initialize:", e)

    # load model
    backend = args.backend
    sampler = args.sampler
    task = args.task
    model_name = benchmark_model
    config = None
    if args.config_file == "true":
        config_file_path = args.config_file_path
        try:
            with open(config_file_path, "r") as f:
                config = json.load(f)
                print(f"config file {config_file_path} is loaded")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"config file {config_file_path} is not loaded: {e}")
    architecture = "cpu" if args.use_gpu == 0 else "gpu"
    trace_level = args.trace_level
    gpu_trace = (
        True
        if (TRACE_LEVEL.index(trace_level) >= TRACE_LEVEL.index("SYSTEM_LIBRARY_TRACE"))
        and (args.gpu_trace == "true")
        else False
    )
    security_check = True if args.security_check == "true" else False

    save_trace_result = (
        True
        if (args.save_trace_result == "true") and (trace_level != "NO_TRACE")
        else False
    )
    save_trace_result_path = args.save_trace_result_path if save_trace_result else None
    print(save_trace_result_path, "save trace result path")
    tracer, root_span, ctx = Tracer.create(
        trace_level=trace_level,
        sampler_type=sampler,
        sampling_ratio=args.sampler_ratio,
        save_trace_result_path=save_trace_result_path,
        endpoint="http://jaeger:4318/v1/traces",
    )
    root_span.set_attribute("cpu_name", get_cpu_name())

    c = None
    print(gpu_trace, "gpu trace val")
    if (
        architecture == "gpu"
        and gpu_trace
        and tracer.is_trace_enabled("SYSTEM_LIBRARY_TRACE")
    ):
        root_span.set_attribute("gpu_name", get_gpu_name())
        from pycupti import CUPTI

        c = CUPTI(tracer=tracer, runtime_driver_time_adjustment=True)
        print("CUPTI version", c.cuptiGetVersion())

    print(architecture)
    output_processor = OutputProcessor()

    # load backend

    user = "default"
    if backend == "pytorch":
        from mlmodelscope.pytorch_agent import PyTorch_Agent

        agent = PyTorch_Agent(
            task, model_name, architecture, tracer, ctx, security_check, config, user, c
        )
    elif backend == "tensorflow":
        from mlmodelscope.tensorflow_agent import TensorFlow_Agent

        agent = TensorFlow_Agent(
            task, model_name, architecture, tracer, ctx, security_check, config, user
        )
    elif backend == "onnxruntime":
        from mlmodelscope.onnxruntime_agent import ONNXRuntime_Agent

        agent = ONNXRuntime_Agent(
            task, model_name, architecture, tracer, ctx, security_check, config, user
        )
    elif backend == "mxnet":
        from mlmodelscope.mxnet_agent import MXNet_Agent

        agent = MXNet_Agent(
            task, model_name, architecture, tracer, ctx, security_check, config, user
        )
    elif backend == "jax":
        from mlmodelscope.jax_agent import JAX_Agent

        agent = JAX_Agent(
            task, model_name, architecture, tracer, ctx, security_check, config, user
        )
    else:
        raise NotImplementedError(f"{backend} agent is not supported")

    final_results = {
        "runtime": backend,
        "time": int(time.time()),
        # "args": vars(args),
        # "cmdline": str(args),
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

    # don't include warmup traces
    if not include_warmup_trace:
        tracer.set_trace_level("NO_TRACE")

    # warmup
    dataset.load([0])
    for _ in range(5):
        img = dataset.get_samples([0])
        # _ = backend.predict({backend.inputs[0]: img})
        agent.predict(0, DataLoader(img, args.max_batchsize), output_processor)
    dataset.unload(None)

    # continue with traces
    if not include_warmup_trace:
        tracer.set_trace_level(trace_level)

    scenario = SCENARIO_MAP[arg_scenario]

    # for lg.ConstructSUT()
    def issue_queries(query_samples):
        idx = np.array([q.index for q in query_samples]).astype(np.int32)
        query_id = [q.id for q in query_samples]
        if args.dataset == "brats2019":
            start = time.time()
            response_array_refs = []
            response = []
            for index, qid in enumerate(query_id):
                # processed_results = so.IssueQuery(1, idx[i][np.newaxis])
                processed_results = agent.predict(
                    0,
                    DataLoader(
                        dataset.get_samples(idx[index][np.newaxis]), args.max_batchsize
                    ),
                    output_processor,
                    mlharness=True,
                )
                # processed_results = json.loads(processed_results.decode('utf-8'))
                for j in range(len(processed_results[index])):
                    processed_results[index][j] = [idx[index]] + processed_results[
                        index
                    ][j]
                response_array = array.array(
                    "B",
                    np.array(idx[index] + processed_results[0], np.float16).tobytes(),
                )
                response_array_refs.append(response_array)
                bi = response_array.buffer_info()
                response.append(lg.QuerySampleResponse(qid, bi[0], bi[1]))
            result_timeing.append(time.time() - start)
            lg.QuerySamplesComplete(response)
        else:
            start = time.time()

            # processed_results = so.IssueQuery(len(idx), idx)
            processed_results = agent.predict(
                0,
                DataLoader(dataset.get_samples(idx), args.max_batchsize),
                output_processor,
                mlharness=True,
            )

                
            result_timeing.append(time.time() - start)
            # processed_results = json.loads(processed_results.decode('utf-8'))
            response_array_refs = []
            response = []
            for index, qid in enumerate(query_id):
                if args.dataset == "coco":
                    for j in range(len(processed_results[index])):
                        processed_results[index][j] = [idx[index]] + processed_results[
                            index
                        ][j]
                dtype = np.int64 if args.dataset == "cnn" else np.float32
                response_array = array.array(
                    "B", np.array(processed_results[index], dtype).tobytes()
                )
                response_array_refs.append(response_array)
                bi = response_array.buffer_info()
                response.append(lg.QuerySampleResponse(qid, bi[0], bi[1]))



            lg.QuerySamplesComplete(response)

    # for lg.ConstructSUT()
    def flush_queries():
        pass

    settings = lg.TestSettings()

    if mlperf_model_name:
        settings.FromConfig(mlperf_conf, mlperf_model_name, arg_scenario)
        settings.FromConfig(user_conf, mlperf_model_name, arg_scenario)

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

    # if args.accuracy:
    #     accuracy_script_paths = {'coco': os.path.realpath('./inference/vision/classification_and_detection/tools/accuracy-coco.py'),
    #                     'imagenet': os.path.realpath('./inference/vision/classification_and_detection/tools/accuracy-imagenet.py'),
    #                     'squad': os.path.realpath('./inference/language/bert/accuracy-squad.py'),
    #                     'brats2019': os.path.realpath('./inference/vision/medical_imaging/3d-unet/accuracy-brats.py'),
    #                     'cnn': os.path.realpath('./inference/language/gpt-j/evaluation.py')}
    #     accuracy_script_path = accuracy_script_paths[args.dataset]
    #     accuracy_file_path = os.path.join(log_dir, 'mlperf_log_accuracy.json')

    data_dir = os.environ["DATA_DIR"]

    result_dict = calculate_accuracy(
        f"{log_dir}/mlperf_log_accuracy.json",
        f"{data_dir}/val_map.txt",
        scenario=arg_scenario,
    )

    print(f"last_timeing len: {len(last_timeing)}")
    add_results(
        final_results,
        "{}".format(scenario),
        result_dict,
        last_timeing,
        time.time() - dataset.last_loaded,
        args.accuracy,
    )

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)
    return final_results


def main():
    args = get_args()
    log.info(args)
    benchmark_results = []
    tracer_results = []
    trace_result_path = "trace_result.txt"


    args.top_k_avg_time_consuming_kernels = (
        args.top_k_avg_time_consuming_kernels == "true"
    )
    args.top_k_total_time_consuming_kernels = (
        args.top_k_total_time_consuming_kernels == "true"
    )
    args.top_k_total_time_consuming_kernels_with_count = (
        args.top_k_total_time_consuming_kernels_with_count == "true"
    )
    args.average_time_for_kernel = args.average_time_for_kernel == "true"
    args.time_consuming_layers_by_depth = args.time_consuming_layers_by_depth == "true"
    args.average_time_for_layer = args.average_time_for_layer == "true"
    args.find_kernels_to_optimize = args.find_kernels_to_optimize == "true"
    args.layerwise_kernel = args.layerwise_kernel == "true"

    if args.mlperf_model_names:
        for model_name, mlperf_model_name, scenario_name in zip(
            args.model_names, args.mlperf_model_names, args.scenario
        ):
            if mlperf_model_name == "None":
                mlperf_model_name = None
            summary_result = run_harness(args, model_name, scenario_name, mlperf_model_name)
            benchmark_results.append(summary_result)

            with open(trace_result_path, "r") as f:
                trace_result_raw_text = f.read()
                trace_result = [
                    json.loads(obj)
                    for obj in trace_result_raw_text.split("\n\n")
                    if obj
                ]

            spans = trace_result

            print(len(spans), "len of spans")

            if args.compare_trace == "true":
                tracer_results.append(
                    analyze_trace_results(
                        request_data=spans,
                        top_k_avg_time_consuming_kernels=args.top_k_avg_time_consuming_kernels,
                        top_k_total_time_consuming_kernels=args.top_k_total_time_consuming_kernels,
                        top_k_total_time_consuming_kernels_with_count=args.top_k_total_time_consuming_kernels_with_count,
                        average_time_for_kernel=args.average_time_for_kernel,
                        time_consuming_layers_by_depth=args.time_consuming_layers_by_depth,
                        average_time_for_layer=args.average_time_for_layer,
                        find_kernels_to_optimize=args.find_kernels_to_optimize,
                        layerwise_kernel=args.layerwise_kernel,
                    )
                )


    comparison_results, scenario_results = compare_loadgen_results(benchmark_results, args.model_names)

    
    if args.save_plots == "true":
        visualize_latency_model_comparison(scenario_results)

    if args.compare_trace == "true":
        compare_trace_res = compare_analysis_results(
            tracer_results,
            args.model_names,
        )
        display_comparison_results(compare_trace_res)


    # Replace with your actual output file path


if __name__ == "__main__":
    main()
