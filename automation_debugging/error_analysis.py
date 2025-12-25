import json
import csv
import os
import time
from tabulate import tabulate
from dotenv import load_dotenv
import google.generativeai as genai
from collections import defaultdict

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

# Use a model that supports JSON mode if possible, or just standard Gemini 2.5 Flash
# Assuming 'gemini-2.5-flash' is available and good for this.
MODEL_NAME = 'gemini-2.5-flash-lite'

def analyze_errors_with_gemini(task_name, errors_dict, known_categories=None):
    """
    Sends error logs to Gemini to categorize and count them.
    Passes previously seen categories to encourage consistency.
    """
    if not errors_dict:
        return {}
        
    known_cats_str = ", ".join(known_categories) if known_categories else "None"

    # Prepare prompt
    prompt = f"""
    You are an expert software debugger. I have a list of error logs for the task '{task_name}'.
    Please analyze them and group them by the root cause of the failure.
    
    Current known error categories from previous tasks: [{known_cats_str}]
    If an error fits one of these known categories, USE THAT EXACT CATEGORY NAME.
    If it is a new type of error, create a new concise category name.
    
    Here are the errors (Model Name -> Error Log):
    {json.dumps(errors_dict, indent=2)}
    
    Return ONLY a valid JSON object in the following format:
    {{
        "error_summary": {{
            "Category Name": count,
            ...
        }}
    }}
    """

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        
        # Clean up response if it contains markdown code blocks
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        
        result = json.loads(text)
        return result.get("error_summary", {})
    except Exception as e:
        print(f"Error calling Gemini API for {task_name}: {e}")
        return {"API Error": len(errors_dict)}

def main():
    input_file = "error_data.json"
    if not os.path.exists(input_file):
        print(f"File {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        data = json.load(f)

    table_data = []
    
    print(f"Analyzing {len(data)} tasks using {MODEL_NAME}...\n")
    
    known_categories = set()

    for task, stats in data.items():
        if task in open("error_summary.csv").read():
            continue  # Skip if task already processed
        failures = stats.get("failures", 0)
        successes = stats.get("successes", 0)
        errors = stats.get("errors", {})
        
        error_summary_str = ""
        
        if failures > 0 and errors:
            print(f"Processing errors for task: {task} ({failures} failures)...")
            
            # Call Gemini
            summary = analyze_errors_with_gemini(task, errors, known_categories=list(known_categories))
            
            # Update known categories with any new ones found
            if summary:
                known_categories.update(summary.keys())
            
            # Format summary for the table
            summary_lines = [f"{k}: {v}" for k, v in summary.items()]
            error_summary_str = "\n".join(summary_lines)
            
            # Be nice to the API rate limits if necessary
            time.sleep(1) 
        elif failures > 0:
            error_summary_str = "No error details found."
        else:
            error_summary_str = "-"

        table_data.append([task, successes, failures, error_summary_str])


    # Write table to file
    headers = ["Task", "Successes", "Failures", "Error Analysis"]
    output_filename = "error_summary.csv"
    with open(output_filename, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table_data)
    
    print(f"\nAnalysis complete. Results written to {output_filename}")

if __name__ == "__main__":
    main()
