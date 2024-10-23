from tabulate import tabulate

def process_benchmark_results(benchmark_results):
    metrics = {}

    metric_labels = [
        "Scenario",
        "Accuracy (%)",
        "QPS",
        "Mean Latency (s)",
        "P50 Latency (s)",
        "P90 Latency (s)",
        "P99 Latency (s)",
        "Total Time (s)",
        "Total Queries",
        "Good Items",
        "Total Items",
    ]

    # Initialize metrics dictionary
    for label in metric_labels:
        metrics[label] = []

    base_arg = benchmark_results[0].get('args', {})
    model_names = base_arg.get('model_names', [])

    best_accuracy = [0, ""]
    best_qps = [0, ""]
    best_mean_latency = [0, ""]
    best_p50_latency = [0, ""]
    best_p90_latency = [0, ""]
    best_p99_latency = [0, ""]

    for index, result in enumerate(benchmark_results):
        args = result.get("args", {})
        test_scenario = result.get("TestScenario.SingleStream", {})
        scenario = args.get("scenario", "N/A")
        accuracy = test_scenario.get("accuracy", "N/A")
        qps = test_scenario.get("qps", "N/A")
        mean_latency = test_scenario.get("mean", "N/A")
        percentiles = test_scenario.get("percentiles", {})
        p50 = percentiles.get("50.0", "N/A")
        p90 = percentiles.get("90.0", "N/A")
        p99 = percentiles.get("99.0", "N/A")
        took = test_scenario.get("took", "N/A")
        count = test_scenario.get("count", "N/A")
        good_items = test_scenario.get("good_items", "N/A")
        total_items = test_scenario.get("total_items", "N/A")

        # Update best vars

        if accuracy != "N/A" and accuracy > float(best_accuracy[0]) or best_accuracy[1] == "":
            best_accuracy[0] = accuracy
            best_accuracy[1] = model_names[index]
        if qps != "N/A" and qps > float(best_qps[0]) or best_qps[1] == "":
            best_qps[0] = qps
            best_qps[1] = model_names[index]        
        if mean_latency != "N/A" and mean_latency > float(best_mean_latency[0]) or best_mean_latency[1] == "":
            best_mean_latency[0] = mean_latency
            best_mean_latency[1] = model_names[index]      
        if p50 != "N/A" and p50 > float(best_p50_latency[0]) or best_p50_latency[1] == "":
            best_p50_latency[0] = p50
            best_p50_latency[1] = model_names[index]  
        if p90 != "N/A" and p90 > float(best_p90_latency[0]) or best_p90_latency[1] == "":
            best_p90_latency[0] = p90
            best_p90_latency[1] = model_names[index]  
        if p99 != "N/A" and p99 > float(best_p99_latency[0]) or best_p99_latency[1] == "":
            best_p99_latency[0] = p99
            best_p99_latency[1] = model_names[index]

        metrics["Scenario"].append(scenario)
        metrics["Accuracy (%)"].append(accuracy)
        metrics["QPS"].append(qps)
        metrics["Mean Latency (s)"].append(mean_latency)
        metrics["P50 Latency (s)"].append(p50)
        metrics["P90 Latency (s)"].append(p90)
        metrics["P99 Latency (s)"].append(p99)
        metrics["Total Time (s)"].append(took)
        metrics["Total Queries"].append(count)
        metrics["Good Items"].append(good_items)
        metrics["Total Items"].append(total_items)

    # Prepare table data
    table_data = []
    for label in metric_labels:
        row = [label] + metrics[label]
        table_data.append(row)

    headers = ["Metric"] + model_names

    print("\Benchmark Results:")

    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Basic insights based on the data
    print("\nBasic Insights:")

    table_data = [
        ["Best Accuracy", best_accuracy[0], best_accuracy[1]],
        ["Best QPS", best_qps[0], best_qps[1]],
        ["Best Mean Latency", best_mean_latency[0], best_mean_latency[1]],
        ["Best P50 Latency", best_p50_latency[0], best_p50_latency[1]],
        ["Best P90 Latency", best_p90_latency[0], best_p90_latency[1]],
        ["Best P99 Latency", best_p99_latency[0], best_p99_latency[1]],
    ]

    print(
        tabulate(
            table_data, headers=["Metric", "Value", "Best Model"], tablefmt="pretty"
        )
    )


    # for i, model_name in enumerate(model_names):
    #     accuracy = metrics['Accuracy (%)'][i]
    #     qps = metrics['QPS'][i]
    #     mean_latency = metrics['Mean Latency (s)'][i]
    #     good_items = metrics['Good Items'][i]
    #     total_items = metrics['Total Items'][i]
        
    #     print(f"\nModel: {model_name}")
    #     print(f"- Accuracy: {accuracy}% ({good_items}/{total_items} correct predictions)")
    #     print(f"- Throughput (QPS): {qps}")
    #     print(f"- Mean Latency: {mean_latency} seconds")


