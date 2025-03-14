import re 
import json 
from datetime import datetime 
import argparse
import requests
import numpy as np 
import matplotlib.pyplot as plt

def calculate_average(times):
    return sum(times) / len(times) if times else 0

def get_trace_result(trace_result_path):
        with open(trace_result_path, 'r') as f:
            trace_result_raw_text = f.read()
            trace_result = [json.loads(obj) for obj in trace_result_raw_text.split('\n\n') if obj]
        return trace_result

def get_trace_result_analysis(trace_result, gpu=False, layerwise_kernel=False):
    def calculate_duration(start_time, end_time):
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        return (end_dt - start_dt).total_seconds()

    sorted_spans = sorted(trace_result, key=lambda k: k['start_time'])
    print(len(sorted_spans))

    evaluation = False 
    batch_evaluation_times = [] 
    preprocess_times = [] 
    predict_times = [] 
    postprocess_times = [] 

    layer_times = {} 

    kernel_times = {} 

    kernel_for_layer = {} 
    layer_span_id = {} 
    cuda_launch_info = (None, None) # correlation_id, parent_id 

    for span in sorted_spans:
        op_name = span['name'] 
        if op_name.startswith('Evaluate Batch'):
            batch_evaluation_times.append(calculate_duration(span['start_time'], span['end_time']) )
            if not evaluation: 
                evaluation = True
        elif evaluation and op_name.startswith('preprocess'):
            preprocess_times.append(calculate_duration(span['start_time'], span['end_time']) )
        elif evaluation and op_name.startswith('predict'):
            predict_times.append(calculate_duration(span['start_time'], span['end_time']) )
        elif evaluation and op_name.startswith('postprocess'):
            postprocess_times.append(calculate_duration(span['start_time'], span['end_time']) )
        # else if op_name has the following pattern: digit(-digit)*__letter*__letter(s)
        # in detail, the pattern starts with a digit(it could be followed by more digits or letters),
        # ([0-9]+(-[0-9]+)+) 
        # then followed by two underscores, then followed by letter(s) or None 
        # ([A-Za-z0-9]+(\.[A-Za-z0-9]+)+)
        # then followed by two underscores, then followed by letter(s) 
        # ([A-Za-z0-9]+(\.[A-Za-z0-9]+)+) 
        elif evaluation and re.match(r'^\d+[-\d+]*__[\w\.]*__\w+$', op_name):
            # print(op_name)
            if gpu and layerwise_kernel:
                if op_name.startswith('0__'):
                    layer_span_id = {} 
                layer_span_id[span['context']['span_id']] = op_name 
                if op_name not in kernel_for_layer:
                    kernel_for_layer[op_name] = {} 
            
            if op_name not in layer_times:
                layer_times[op_name] = []
            layer_times[op_name].append(calculate_duration(span['start_time'], span['end_time']) ) 
        
        elif evaluation and gpu:
            if op_name.startswith('gpu_kernel'):
                kernel_name = span['attributes']['name'] 
                if kernel_name not in kernel_times:
                    kernel_times[kernel_name] = []
                
                calculated_duration = calculate_duration(span['start_time'], span['end_time']) 

                if layerwise_kernel and cuda_launch_info[0] == span['attributes'].get('correlation_id'):
                    parent_id = cuda_launch_info[1] 
                    parent_layer = layer_span_id.get(parent_id, None) 
                    if parent_layer is not None: 
                        if kernel_name not in kernel_for_layer[parent_layer]:
                            kernel_for_layer[parent_layer][kernel_name] = [] 
                        kernel_for_layer[parent_layer][kernel_name].append(calculated_duration) 

                kernel_times[kernel_name].append(calculated_duration)
            elif layerwise_kernel and op_name.startswith('cuda_launch'):
                correlation_id = span['attributes']['correlation_id'] 
                parent_id = span['parent_id'] 
                cuda_launch_info = (correlation_id, parent_id) 


    num_batch_evaluations = len(batch_evaluation_times)
    avg_batch_evaluation_time = calculate_average(batch_evaluation_times)
    avg_preprocess_time = calculate_average(preprocess_times)
    avg_predict_time = calculate_average(predict_times)
    avg_postprocess_time = calculate_average(postprocess_times)

    return num_batch_evaluations, avg_batch_evaluation_time, avg_preprocess_time, avg_predict_time, avg_postprocess_time, layer_times, kernel_times, kernel_for_layer 

def get_avg_time_for_layer(layer_times, print_layer_times=False): 
    avg_layer_times = {} 
    if print_layer_times:
        print("Layer times: ") 
    for layer, times in layer_times.items():
        avg_layer_times[layer] = calculate_average(times)
        if print_layer_times:
            print(f"{layer}: {avg_layer_times[layer] * 1000} ms") 
    return avg_layer_times 

def get_time_consuming_layers_by_depth(layer_times, top_k_for_print=5, print_layer_times=False, kernel_for_layer=None): 
    depth = 0 
    layer_times_by_depth = {} 
    for layer, times in layer_times.items():
        layer_depth = len(layer.split('__')[0].split('-')) 
        depth = max(depth, layer_depth) 
        if layer_depth not in layer_times_by_depth:
            layer_times_by_depth[layer_depth] = []
        layer_times_by_depth[layer_depth].append((layer, calculate_average(times))) 
    
    if print_layer_times: 
        print(f"Depth: {depth}")

    for layer_depth in range(1, depth + 1):
        # sort the layer times by the average time 
        layer_times_by_depth[layer_depth].sort(key=lambda x: x[1], reverse=True)

        if print_layer_times:
            print(f"Layer depth: {layer_depth}")
            for i, (layer, time) in enumerate(layer_times_by_depth[layer_depth][:top_k_for_print]):
                print(f"{i + 1}: {layer}: {time * 1000} ms")
                if kernel_for_layer is not None:
                    print_kernel_times(get_kernels_for_layer(kernel_for_layer, layer, average=True), average=True, prefix='- ', top_k=3) 
    
    return layer_times_by_depth 

def print_kernel_times(kernel_times, average=True, prefix='', short_kernel_name=True, top_k=5):
    if average: 
        sorted_kernel_times = sorted(kernel_times.items(), key=lambda x: x[1], reverse=True)[:top_k]
    else:
        sorted_kernel_times = sorted(kernel_times.items(), key=lambda x: calculate_average(x[1]), reverse=True)[:top_k]
    for i, (kernel, times) in enumerate(sorted_kernel_times):
        if short_kernel_name: 
            kernel = shorten_kernel_name(kernel) 
        print(f"{prefix}{i + 1}: {kernel}: {times * 1000} ms") 

def get_avg_time_for_kernel(kernel_times, print_kernel_times=False): 
    avg_kernel_times = {} 
    if print_kernel_times:
        print("Kernel times: ") 
    for kernel, times in kernel_times.items():
        avg_kernel_times[kernel] = calculate_average(times)
        if print_kernel_times:
            print(f"{kernel}: {avg_kernel_times[kernel] * 1000} ms") 
    return avg_kernel_times 

def get_top_k_avg_time_consuming_kernels(kernel_times, top_k=5):
    avg_kernel_times = {k: calculate_average(v) for k, v in kernel_times.items()} 
    avg_kernel_times = sorted(avg_kernel_times.items(), key=lambda x: x[1], reverse=True) 
    return avg_kernel_times[:top_k]

def get_top_k_total_time_consuming_kernels(kernel_times, top_k=5):
    total_kernel_times = {k: sum(v) for k, v in kernel_times.items()} 
    total_kernel_times = sorted(total_kernel_times.items(), key=lambda x: x[1], reverse=True) 
    return total_kernel_times[:top_k]

def get_top_k_total_time_consuming_kernels_with_count(kernel_times, top_k=5):
    total_kernel_times = {k: (sum(v), len(v), calculate_average(v)) for k, v in kernel_times.items()} 
    total_kernel_times = sorted(total_kernel_times.items(), key=lambda x: x[1][0], reverse=True) 
    return total_kernel_times[:top_k]

def get_kernels_for_layer(kernel_for_layer, layer_name, average=True):
    all_kernel_times = {} 

    layer_prefix = layer_name.split('__')[0] 

    for layer, kernel_times in kernel_for_layer.items():
        if layer.split('__')[0].startswith(layer_prefix): 
            for kernel, times in kernel_times.items():
                if kernel not in all_kernel_times:
                    all_kernel_times[kernel] = [] 
                all_kernel_times[kernel].extend(times)
    
    if average:
        for kernel, times in all_kernel_times.items():
            all_kernel_times[kernel] = calculate_average(times) 
    
    return all_kernel_times 

def shorten_kernel_name(kernel):
    def remove_pattern(input_string, pattern):
        while re.search(pattern, input_string):
            input_string = re.sub(pattern, '', input_string)
        return input_string
    parentheses = r'\([^()]*\)'  
    angle_brackets = r'<[^<>]*>' 

    kernel = remove_pattern(kernel, parentheses) 
    kernel = remove_pattern(kernel, angle_brackets) 
    if ' ' in kernel:
        kernel = kernel.split(' ')[-1] 
    if '::' in kernel:
        kernel = kernel.split('::')[-1] 
    return kernel 

def get_find_kernels_to_optimize(kernel_times, print_kernels=False, save_fig=False, fig_path='./kernel_performance_metrics.png'):
    metrics = {} 

    for kernel, times in kernel_times.items():
        total_time = sum(times) 
        avg_time = np.mean(times) 
        std_dev = np.std(times) 
        count = len(times) 

        metrics[kernel] = {
            'total_time': total_time,
            'avg_time': avg_time,
            'std_dev': std_dev,
            'count': count
        }

    if print_kernels: 
        # Display the metrics for each kernel
        for kernel, data in metrics.items():
            kernel = shorten_kernel_name(kernel) 
            print(f"Kernel: {kernel}")
            print(f"  Total Time: {data['total_time'] * 1000:.2f} ms")
            print(f"  Average Time: {data['avg_time'] * 1000:.2f} ms")
            print(f"  Standard Deviation: {data['std_dev'] * 1000:.2f} ms")
            print(f"  Invocation Count: {data['count']}")
            print()
    
    # Calculate weighted score for each kernel
    max_total_time = max(data['total_time'] for data in metrics.values())
    max_avg_time = max(data['avg_time'] for data in metrics.values())
    max_count = max(data['count'] for data in metrics.values())
    
    for kernel, data in metrics.items():
        weighted_score = (
            (data['total_time'] / max_total_time) * 0.5 +
            (data['avg_time'] / max_avg_time) * 0.3 +
            (data['count'] / max_count) * 0.2
        )
        metrics[kernel]['weighted_score'] = weighted_score

    # Sort kernels by weighted score
    sorted_kernels = sorted(metrics.items(), key=lambda x: x[1]['weighted_score'], reverse=True)

    # Display sorted kernels
    if print_kernels:
        print("Sorted Kernels by Weighted Score:")
        for kernel, data in sorted_kernels:
            kernel = shorten_kernel_name(kernel) 
            print(f"Kernel: {kernel}, Weighted Score: {data['weighted_score']:.2f}")

    # Pareto Analysis
    total_time_sum = sum(data['total_time'] for data in metrics.values())
    cumulative_time = 0
    pareto_kernels = []

    for kernel, data in sorted_kernels:
        cumulative_time += data['total_time']
        pareto_kernels.append(kernel)
        if cumulative_time / total_time_sum >= 0.8:
            break

    if print_kernels:
        print("\nPareto Kernels (80/20 Rule):")
        for kernel in pareto_kernels:
            # kernel = shorten_kernel_name(kernel) 
            print(kernel)

    if save_fig:
        # Visualization
        # sort kernels keys by weighted score
        kernels = sorted(metrics.keys(), key=lambda x: metrics[x]['weighted_score'], reverse=True) 
        # kernels = pareto_kernels 
        shorten_kernels = [shorten_kernel_name(kernel) for kernel in kernels] 
        total_times = [metrics[kernel]['total_time'] * 1000 for kernel in kernels]
        avg_times = [metrics[kernel]['avg_time'] * 1000 for kernel in kernels]
        counts = [metrics[kernel]['count'] for kernel in kernels]

        plt.figure(figsize=(15, 10))

        plt.subplot(3, 1, 1)
        plt.bar(range(len(shorten_kernels)), total_times, color='b')
        plt.ylabel('Total Time (ms)')
        plt.title('Kernel Performance Metrics')
        plt.xticks([])  # Hide x-ticks for the first plot

        plt.subplot(3, 1, 2)
        plt.bar(range(len(shorten_kernels)), avg_times, color='g')
        plt.ylabel('Average Time (ms)')
        plt.xticks([])  # Hide x-ticks for the first plot

        plt.subplot(3, 1, 3)
        plt.bar(range(len(shorten_kernels)), counts, color='r') 
        plt.ylabel('Invocation Count')
        plt.xlabel('Kernels')
        plt.xticks(range(len(shorten_kernels)), shorten_kernels, rotation=45, ha='right')

        # Adjust layout to make room for the rotated x-axis labels
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
        # Save the figure to a file
        plt.savefig(fig_path)
    
    return pareto_kernels 

def analyze_trace_results(
    trace_result_dir='.',
    trace_result_path='trace_result.txt',
    top_k=5,
    gpu_trace=True,
    layerwise_kernel=True,
    find_kernels_to_optimize=True,
    save_fig=False,
    fig_path='./kernel_performance_metrics.png',
    average_time_for_layer=True,
    time_consuming_layers_by_depth=True,
    average_time_for_kernel=False,
    top_k_avg_time_consuming_kernels=False,
    top_k_total_time_consuming_kernels=False,
    top_k_total_time_consuming_kernels_with_count=False,
    request_data=[]
):

    results = {}

    # trace_result = get_trace_result(trace_result_dir + '/' + trace_result_path)

    trace_result = request_data 
    
    num_batch_evaluations, avg_batch_evaluation_time, avg_preprocess_time, avg_predict_time, avg_postprocess_time, layer_times, kernel_times, kernel_for_layer = get_trace_result_analysis(trace_result, gpu=gpu_trace, layerwise_kernel=layerwise_kernel)
    
    results['num_batch_evaluations'] = num_batch_evaluations
    results['avg_batch_evaluation_time'] = avg_batch_evaluation_time
    results['avg_preprocess_time'] = avg_preprocess_time
    results['avg_predict_time'] = avg_predict_time
    results['avg_postprocess_time'] = avg_postprocess_time
    results['avg_batch_evaluation_time_ms'] = avg_batch_evaluation_time * 1000
    results['avg_preprocess_time_ms'] = avg_preprocess_time * 1000
    results['avg_predict_time_ms'] = avg_predict_time * 1000
    results['avg_postprocess_time_ms'] = avg_postprocess_time * 1000
    
    results['layer_times'] = layer_times
    results['kernel_times'] = kernel_times
    results['kernel_for_layer'] = kernel_for_layer

    if average_time_for_layer:
        results['avg_layer_times'] = get_avg_time_for_layer(layer_times, print_layer_times=False)
    
    if time_consuming_layers_by_depth:
        results['layer_times_by_depth'] = get_time_consuming_layers_by_depth(
            layer_times, top_k_for_print=top_k, print_layer_times=False, kernel_for_layer=kernel_for_layer
        )
    
    if average_time_for_kernel:
        results['avg_kernel_times'] = get_avg_time_for_kernel(kernel_times, print_kernel_times=False)
    
    if top_k_avg_time_consuming_kernels:
        results['top_k_avg_time_consuming_kernels'] = get_top_k_avg_time_consuming_kernels(kernel_times, top_k=top_k)
    
    if top_k_total_time_consuming_kernels:
        results['top_k_total_time_consuming_kernels'] = get_top_k_total_time_consuming_kernels(kernel_times, top_k=top_k)
     
    if top_k_total_time_consuming_kernels_with_count:
        results['top_k_total_time_consuming_kernels_with_count'] = get_top_k_total_time_consuming_kernels_with_count(kernel_times, top_k=top_k)
    
    if find_kernels_to_optimize:
        results['target_kernels'] = get_find_kernels_to_optimize(kernel_times, print_kernels=False, save_fig=save_fig, fig_path=fig_path)
    
    return results


