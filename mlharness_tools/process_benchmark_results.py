from tabulate import tabulate

def process_benchmark_results(benchmark_results):

    model_names = []
    metrics = {}

    metric_labels = [
        'Scenario',
        'Accuracy (%)',
        'QPS',
        'Mean Latency (s)',
        'P50 Latency (s)',
        'P90 Latency (s)',
        'P99 Latency (s)',
        'Total Time (s)',
        'Total Queries',
        'Good Items',
        'Total Items'
    ]

    # Initialize metrics dictionary
    for label in metric_labels:
        metrics[label] = []

    for result in benchmark_results:
        args = result.get('args', {})
        test_scenario = result.get('TestScenario.SingleStream', {})
        model_names_list = args.get('model_names', [])
        model_name = ', '.join(model_names_list) if model_names_list else 'N/A'
        model_names.append(model_name)
        
        # Extract and format metrics
        scenario = args.get('scenario', 'N/A')
        accuracy = f"{test_scenario.get('accuracy', 'N/A'):.2f}"
        qps = f"{test_scenario.get('qps', 'N/A'):.2f}"
        mean_latency = f"{test_scenario.get('mean', 'N/A'):.5f}"
        percentiles = test_scenario.get('percentiles', {})
        p50 = f"{percentiles.get('50.0', 'N/A'):.5f}"
        p90 = f"{percentiles.get('90.0', 'N/A'):.5f}"
        p99 = f"{percentiles.get('99.0', 'N/A'):.5f}"
        took = f"{test_scenario.get('took', 'N/A')}"
        count = test_scenario.get('count', 'N/A')
        good_items = test_scenario.get('good_items', 'N/A')
        total_items = test_scenario.get('total_items', 'N/A')
        
        metrics['Scenario'].append(scenario)
        metrics['Accuracy (%)'].append(accuracy)
        metrics['QPS'].append(qps)
        metrics['Mean Latency (s)'].append(mean_latency)
        metrics['P50 Latency (s)'].append(p50)
        metrics['P90 Latency (s)'].append(p90)
        metrics['P99 Latency (s)'].append(p99)
        metrics['Total Time (s)'].append(took)
        metrics['Total Queries'].append(count)
        metrics['Good Items'].append(good_items)
        metrics['Total Items'].append(total_items)
    
    # Prepare table data
    table_data = []
    for label in metric_labels:
        row = [label] + metrics[label]
        table_data.append(row)

    # Prepare headers
    headers = ['Metric'] + model_names

    # Print the table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Basic insights based on the data
    print("\nBasic Insights:")

    for i, model_name in enumerate(model_names):
        accuracy = metrics['Accuracy (%)'][i]
        qps = metrics['QPS'][i]
        mean_latency = metrics['Mean Latency (s)'][i]
        good_items = metrics['Good Items'][i]
        total_items = metrics['Total Items'][i]
        
        print(f"\nModel: {model_name}")
        print(f"- Accuracy: {accuracy}% ({good_items}/{total_items} correct predictions)")
        print(f"- Throughput (QPS): {qps}")
        print(f"- Mean Latency: {mean_latency} seconds")
        if float(accuracy) < 70:
            print("- Insight: The model has low accuracy. Consider improving the model or dataset.")
        if float(qps) < 50:
            print("- Insight: The throughput is low. Optimization may be needed.")
        if float(mean_latency) > 0.02:
            print("- Insight: The latency is higher than expected for a SingleStream scenario.")

