from collections import defaultdict
import statistics
import json
import pandas as pd
from tabulate import tabulate
from datetime import datetime
import math


def parse_trace_output(filename):
    spans = []

    with open(filename, 'r') as file:
        content = file.read()
        span_entries = content.strip().split('\n\n')

    for span_entry in span_entries:
        if not span_entry.strip():
            continue  # Skip empty entries
        try:
            span_data = json.loads(span_entry)
            spans.append(span_data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            continue

    return spans



def extract_span_info(spans):
    # Nested defaultdict to group spans by trace_level and then by name
    grouped_spans = defaultdict(lambda: defaultdict(list))

    proccessed_spans = {
        "total_spans": len(spans),
        "grouped_spans": grouped_spans,  # Store grouped spans here
    }

    for span in spans:
        # Extract attributes
        attributes = span.get('attributes', {})
        trace_level = attributes.get('trace_level', 'UNKNOWN')
        name = span.get('name', 'Unnamed Operation')

        # Parse start and end times
        start_time_str = span.get('start_time')
        end_time_str = span.get('end_time')

        try:
            start_time = datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M:%S.%fZ')
            end_time = datetime.strptime(end_time_str, '%Y-%m-%dT%H:%M:%S.%fZ')
            # Convert to Unix timestamps in nanoseconds
            start_time_ns = int(start_time.timestamp() * 1e9)
            end_time_ns = int(end_time.timestamp() * 1e9)
        except Exception as e:
            print(f"Error parsing times for span {span.get('name')}: {e}")
            continue

        # Calculate duration in milliseconds
        duration_ms = (end_time_ns - start_time_ns) / 1e6

        # Store the span info
        span_info = {
            'name': name,
            'attributes': attributes,
            'start_time': start_time_str,
            'end_time': end_time_str,
            'duration_ms': duration_ms
        }

        # Group the spans
        grouped_spans[trace_level][name].append(span_info)

    return proccessed_spans


def compute_percentile(durations, percentile):
    index = (len(durations) - 1) * (percentile / 100)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return durations[int(index)]
    else:
        return durations[lower] * (upper - index) + durations[upper] * (index - lower)


def single_run_insights(insights):

    # Dictionary to store the computed statistics
    stats = defaultdict(lambda: defaultdict(dict))

    grouped_spans = insights.get('grouped_spans', {})

    for trace_level, names_dict in grouped_spans.items():
        for name, spans in names_dict.items():
            durations = [span['duration_ms'] for span in spans]
            durations.sort()  # Sort durations for percentile calculations

            avg_duration = sum(durations) / len(durations)
            min_duration = durations[0]
            max_duration = durations[-1]
            count = len(durations)

            # Standard Deviation
            variance = sum((x - avg_duration) ** 2 for x in durations) / count
            std_deviation = math.sqrt(variance)

            # Percentiles
            p50 = compute_percentile(durations, 50)
            p90 = compute_percentile(durations, 90)
            p95 = compute_percentile(durations, 95)
            p99 = compute_percentile(durations, 99)

            total_duration = sum(durations)

            # Store the computed statistics
            stats[trace_level][name] = {
                'average_duration_ms': avg_duration,
                'min_duration_ms': min_duration,
                'max_duration_ms': max_duration,
                'std_deviation_ms': std_deviation,
                'p50_duration_ms': p50,
                'p90_duration_ms': p90,
                'p95_duration_ms': p95,
                'p99_duration_ms': p99,
                'total_duration_ms': total_duration,
                'count': count
            }

    return stats


def generate_tables_and_plots(stats_list, model_names):
    
    all_trace_levels = set()
    for stats in stats_list:
        all_trace_levels.update(stats.keys())

    # Map model names to their corresponding stats
    model_stats_map = dict(zip(model_names, stats_list))

    # For each trace_level, create a table for each model
    for trace_level in sorted(all_trace_levels):
        print(f"\nTrace Level: {trace_level}")
        for model_name in model_names:
            stats = model_stats_map.get(model_name, {})
            if trace_level not in stats:
                continue  # Skip if the model doesn't have this trace_level
            print(f"\nModel: {model_name}")
            # Get all operation names for this trace_level in this model
            operation_names = stats[trace_level].keys()

            # Prepare table headers
            headers = [
                'Operation',
                'Avg (ms)',
                'Min (ms)',
                'Max (ms)',
                'StdDev (ms)',
                'P50 (ms)',
                'P90 (ms)',
                'P95 (ms)',
                'P99 (ms)',
                'Total Dur (ms)',
                'Count'
            ]

            # Prepare table rows
            table_rows = []
            for operation_name in sorted(operation_names):
                op_stats = stats[trace_level][operation_name]
                row = [
                    operation_name,
                    f"{op_stats['average_duration_ms']:.2f}",
                    f"{op_stats['min_duration_ms']:.2f}",
                    f"{op_stats['max_duration_ms']:.2f}",
                    f"{op_stats['std_deviation_ms']:.2f}",
                    f"{op_stats['p50_duration_ms']:.2f}",
                    f"{op_stats['p90_duration_ms']:.2f}",
                    f"{op_stats['p95_duration_ms']:.2f}",
                    f"{op_stats['p99_duration_ms']:.2f}",
                    f"{op_stats['total_duration_ms']:.2f}",
                    f"{op_stats['count']}"
                ]
                table_rows.append(row)

            # Print the table using tabulate
            print(tabulate(table_rows, headers=headers, tablefmt='grid'))


