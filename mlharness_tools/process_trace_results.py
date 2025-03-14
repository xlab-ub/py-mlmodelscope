from collections import defaultdict
from tabulate import tabulate


def get_top_metric(metric_list):
    max_metric = max(metric_list, default=None)

    if max_metric is None:
        return (0,0)

    max_metric_index = metric_list.index(max_metric)

    return (max_metric_index, max_metric)


def compare_analysis_results(trace_results, model_names, gpu_trace = True ):
    """
    Arguments:
        trace_results: List of dictionaries containing analysis results from analyze_trace_results
        metrics_to_compare : List of metrics to compare. If None, compare standard metrics.
        gpu_trace: Boolean indicating if there is a GPU trace
    """
    # can add metrics_to_compare later for more flexibility
    # units in microsce seconds

    metrics_to_compare = defaultdict(list)

    basic_metrics = [
        "avg_batch_evaluation_time",
        "avg_preprocess_time",
        "avg_predict_time",
        "avg_postprocess_time",
    ]

    # populate metrics_to_compare
    for metric in basic_metrics:
        metrics_to_compare[metric]

    # if gpu_trace:
    #     metrics_to_compare["avg_layer_times"]
    #     metrics_to_compare["avg_kernel_times"]

    # populate metrics_to_compare with the results from the trace_results
    for trace in trace_results:
        for metric in trace:
            if metric in metrics_to_compare:
                metrics_to_compare[metric].append(trace[metric])

    comparison = {}
    top_avg_metrics = {}

    for metric in metrics_to_compare:
        top_metric = get_top_metric(metrics_to_compare[metric])
        top_avg_metrics[f"top_{metric}"] = f"{model_names[top_metric[0]]}: {top_metric[1]}"

    comparison["basic_comparison"] = top_avg_metrics
    return comparison

def compare_loadgen_results(loadgen_results, model_names):
    if not loadgen_results:
        return {}
    
    # Dictionary to organize results
    all_entries = []
    comparison = {}
    scenario_results = {}

    # Process each benchmark result
    for i, result in enumerate(loadgen_results):
        # Get the specific model for this result based on its position in the results list
        model_index = i % len(model_names) if model_names else 0
        model_name = model_names[model_index] if model_index < len(model_names) else "Unknown Model"
        
        # Look for scenario results in the result dictionary
        for key, value in result.items():
            scenario_name = None
            
            # Handle TestScenario.X format (from lg.TestScenario enum)
            if key.startswith('TestScenario.'):
                scenario_name = key.split('.')[-1]
            # Direct scenario name format
            elif key in ["SingleStream", "MultiStream", "Server", "Offline"]:
                scenario_name = key
            
            if scenario_name and isinstance(value, dict):
                if model_name not in scenario_results:
                    scenario_results[model_name] = []
                
                # Collect relevant metrics - convert to ms for better readability
                entry = {
                    'scenario': scenario_name,
                    'mean_latency_ms': round(value.get('mean', 0) * 1000, 3),
                    'qps': round(value.get('qps', 0), 2),
                    'accuracy': value.get('accuracy', 0),
                    'p50_latency_ms': round(value.get('percentiles', {}).get('50.0', 0) * 1000, 3),
                    'p80_latency_ms': round(value.get('percentiles', {}).get('80.0', 0) * 1000, 3),
                    'p90_latency_ms': round(value.get('percentiles', {}).get('90.0', 0) * 1000, 3),
                    'p95_latency_ms': round(value.get('percentiles', {}).get('95.0', 0) * 1000, 3),
                    'p99_latency_ms': round(value.get('percentiles', {}).get('99.0', 0) * 1000, 3),
                    'p99_9_latency_ms': round(value.get('percentiles', {}).get('99.9', 0) * 1000, 3)
                }
                all_entries.append({**entry, 'model': model_name})
                scenario_results[model_name].append(entry)
    
    # Print results as tables
    for model_name, entries in scenario_results.items():
        print(f"\n{model_name} - Results:")
        headers = ["Scenario", "Mean Latency (ms)", "QPS", "Accuracy (%)", 
                  "P50 Latency", "P80 Latency", "P90 Latency", 
                  "P95 Latency", "P99 Latency", "P99.9 Latency"]
        table_data = []
        for entry in entries:
            table_data.append([
                entry['scenario'],
                entry['mean_latency_ms'],
                entry['qps'],
                entry['accuracy'],
                entry['p50_latency_ms'],
                entry['p80_latency_ms'],
                entry['p90_latency_ms'],
                entry['p95_latency_ms'],
                entry['p99_latency_ms'],
                entry['p99_9_latency_ms']
            ])
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Create a single table showing best model for each metric
    if all_entries:
        print("\nBest Results Across All Scenarios:")
        best_metrics = {
            'Mean Latency (ms)': min(all_entries, key=lambda x: x['mean_latency_ms']),
            'QPS': max(all_entries, key=lambda x: x['qps']),
            'Accuracy (%)': max(all_entries, key=lambda x: x['accuracy']),
            'P50 Latency': min(all_entries, key=lambda x: x['p50_latency_ms']),
            'P80 Latency': min(all_entries, key=lambda x: x['p80_latency_ms']),
            'P90 Latency': min(all_entries, key=lambda x: x['p90_latency_ms']),
            'P95 Latency': min(all_entries, key=lambda x: x['p95_latency_ms']),
            'P99 Latency': min(all_entries, key=lambda x: x['p99_latency_ms']),
            'P99.9 Latency': min(all_entries, key=lambda x: x['p99_9_latency_ms'])
        }
        
        # Create summary table
        summary_headers = ["Metric", "Best Model", "Value", "Scenario"]
        summary_table = []
        for metric_name, entry in best_metrics.items():
            value = None
            if metric_name == 'Mean Latency (ms)':
                value = f"{entry['mean_latency_ms']} ms"
            elif metric_name == 'QPS':
                value = f"{entry['qps']} qps"
            elif metric_name == 'Accuracy (%)':
                value = f"{entry['accuracy']}%"
            elif metric_name == 'P50 Latency':
                value = f"{entry['p50_latency_ms']} ms"
            elif metric_name == 'P80 Latency':
                value = f"{entry['p80_latency_ms']} ms"
            elif metric_name == 'P90 Latency':
                value = f"{entry['p90_latency_ms']} ms"
            elif metric_name == 'P95 Latency':
                value = f"{entry['p95_latency_ms']} ms"
            elif metric_name == 'P99 Latency':
                value = f"{entry['p99_latency_ms']} ms"
            elif metric_name == 'P99.9 Latency':
                value = f"{entry['p99_9_latency_ms']} ms"
            
            summary_table.append([
                metric_name,
                entry['model'],
                value,
                entry['scenario']
            ])
        
        print(tabulate(summary_table, headers=summary_headers, tablefmt="grid"))
        
        # Store the best metrics in the comparison dictionary
        comparison['best_overall'] = best_metrics
    
    # Store the complete data for visualization
    comparison['all_entries'] = all_entries
    comparison['scenario_results'] = scenario_results
    
    return comparison, scenario_results


def display_comparison_results(comparison_results):
    
    if 'basic_comparison' in comparison_results:
        basic_comp = comparison_results['basic_comparison']
        
        headers = ["Metric", "Best Model", "Value"]
        table_data = []
        
        for metric, value in basic_comp.items():
            model, time_value = value.split(': ')
            clean_metric = metric.replace('top_', '')
            table_data.append([clean_metric, model, time_value])
        
        print("\nBasic Trace Comparison Results:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))


