import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def visualize_latency_model_comparison(data, save_path="latency_comparison", normalize_spider=False):
    """
    data : dict { key:  scenario_name , value: scenario_results }
    save_path : folder_name, will be created if it doesn't exist
    """
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    metrics = ['mean_latency_ms', 'p50_latency_ms', 'p80_latency_ms', 
               'p90_latency_ms', 'p95_latency_ms', 'p99_latency_ms', 'p99_9_latency_ms']

    display_names = {
        'mean_latency_ms': 'Mean Latency', 
        'p50_latency_ms': 'P50 Latency', 
        'p80_latency_ms': 'P80 Latency',
        'p90_latency_ms': 'P90 Latency', 
        'p95_latency_ms': 'P95 Latency', 
        'p99_latency_ms': 'P99 Latency',
        'p99_9_latency_ms': 'P99.9 Latency'
    }

    # Create color cycle for multiple models
    colors = plt.cm.tab10.colors
    
    # Extract values for each model
    model_values = {}
    model_scenarios = {}
    
    for i, (model_name, model_data) in enumerate(data.items()):
        model_values[model_name] = [model_data[0][m] for m in metrics]
        model_scenarios[model_name] = model_data[0]['scenario']
    
    # --- SPIDER CHART ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)

    N = len(metrics)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  
    
    # Normalize 
    if normalize_spider:
        # Find max value for each metric across all models
        max_metric_values = {m: max(model_data[0][m] for model_name, model_data in data.items()) 
                            for m in metrics}
        
        # Normalize each model's values
        normalized_values = {}
        for model_name, values in model_values.items():
            normalized_values[model_name] = [
                model_values[model_name][i] / max_metric_values[metrics[i]]
                for i in range(len(metrics))
            ]
        plotting_values = normalized_values
        value_suffix = "" # No ms for normalized values
        title_suffix = "(normalized values)"
    else:
        plotting_values = model_values
        value_suffix = "ms"
        title_suffix = "(lower is better)"

    # Plot each model
    for i, (model_name, values) in enumerate(plotting_values.items()):
        values_closed = values + values[:1]  
        ax.plot(angles, values_closed, 'o-', linewidth=2, color=colors[i % len(colors)], 
                label=f'{model_name} ({model_scenarios[model_name]})')
        ax.fill(angles, values_closed, alpha=0.1, color=colors[i % len(colors)])

    # Add metric labels
    metric_labels = [display_names[m] for m in metrics]
    plt.xticks(angles[:-1], metric_labels, size=12)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Adjust y-axis
    if normalize_spider:
        plt.ylim(0, 1.1)
    else:
        max_value = max([max(vals) for vals in plotting_values.values()]) * 1.1
        plt.ylim(0, max_value)

    plt.title(f'Model Latency Comparison {title_suffix}', size=15)

    # Add value annotations (if not too many models)
    if len(plotting_values) <= 3:
        for i, (model_name, values) in enumerate(plotting_values.items()):
            for j, value in enumerate(values):
                y_offset = 0.05 if normalize_spider else max_value * 0.03
                if normalize_spider:
                    value_text = f'{value:.2f}'
                else:
                    value_text = f'{value:.1f}{value_suffix}'
                    
                ax.annotate(value_text, 
                            (angles[j], values[j]),
                            xytext=(angles[j], values[j] + y_offset),
                            fontsize=8)

    # Add ms markers on radial lines
    if normalize_spider:
        radial_ticks = np.arange(0, 1.1, 0.2)
    else:
        radial_ticks = np.arange(0, max_value, max_value/5)
    ax.set_rticks(radial_ticks)
    ax.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax.grid(True)

    plt.tight_layout()
    if save_path:
        spider_filename = "spider"
        if normalize_spider:
            spider_filename += "_normalized"
        output_path = os.path.join(save_path, spider_filename + ".png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # --- BAR CHART ---
    plt.figure(figsize=(14, 8))

    # Set up bar positions
    num_models = len(model_values)
    bar_width = 0.8 / num_models
    x = np.arange(len(metrics))
    
    # We always use raw values for the bar chart
    max_raw_value = max([max(vals) for vals in model_values.values()]) * 1.1

    # Create bars for each model
    for i, (model_name, values) in enumerate(model_values.items()):
        position = x - 0.4 + bar_width/2 + i*bar_width
        bars = plt.bar(position, values, bar_width, 
                       label=f'{model_name} ({model_scenarios[model_name]})',
                       color=colors[i % len(colors)])
        
        # Add value annotations
        for j, v in enumerate(values):
            plt.text(position[j], v + max_raw_value * 0.01, f'{v:.1f}', 
                     ha='center', fontsize=9, rotation=90 if v < max_raw_value * 0.05 else 0)

    # Add labels and title
    plt.xlabel('Performance Metrics', fontsize=14)
    plt.ylabel('Latency (ms)', fontsize=14)
    plt.title('Model Latency Comparison (lower is better)', fontsize=16)
    plt.xticks(x, metric_labels, rotation=45)
    plt.legend()

    plt.tight_layout()

    if save_path:
        output_path = os.path.join(save_path, "bar_chart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')



