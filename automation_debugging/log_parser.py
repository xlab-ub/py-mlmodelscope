import json
import os

def parse_master_log(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    data_summary = {}

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines or just brackets indicating start/end of list
            if not line or line == '[' or line == ']':
                continue
            
            # Remove trailing comma if present (common in manually constructed JSON lists)
            if line.endswith(','):
                line = line[:-1]
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # print(f"Warning: Could not decode line {line_num}: {line[:50]}...")
                continue

            # Expecting format: {"task_name": {"model_name": {"failures": int, "successes": int, "error": str}}}
            for task, models in entry.items():
                if task in open(output_file).read():
                    continue  # Skip if task already processed
                if task not in data_summary:
                    data_summary[task] = {
                        "failures": 0,
                        "successes": 0,
                        "errors": {}
                    }
                
                for model_name, stats in models.items():
                    failures = stats.get("failures", 0)
                    successes = stats.get("successes", 0)
                    error = stats.get("error")

                    data_summary[task]["failures"] += failures
                    data_summary[task]["successes"] += successes
                    
                    if error:
                        # Trim error to last 3000 chars
                        trimmed_error = error[-3000:] if len(error) > 3000 else error
                        data_summary[task]["errors"][model_name] = trimmed_error

    # Write the summary to the output file
    try:
        with open(output_file, 'a') as out_f:
            json.dump(data_summary, out_f, indent=4)
        print(f"Successfully created {output_file}")
    except IOError as e:
        print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    INPUT_FILE = "master_log.json"
    OUTPUT_FILE = "error_data.json"
    parse_master_log(INPUT_FILE, OUTPUT_FILE)