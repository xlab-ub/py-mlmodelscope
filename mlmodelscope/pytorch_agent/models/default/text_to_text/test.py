import subprocess
import json
import shlex
from pathlib import Path
import time
from datetime import datetime
import re
import subprocess
import sys
import os

# ! This script installs depencies by itself which could be a security risk !


def install_packages_in_conda(package_names):
    """
    Installs a list of Python packages using the pip associated with the
    current Python executable (expected to be within a Conda environment).

    Args:
        package_names (list or str): A string for a single package,
                                     or a list of strings for multiple packages.
    """

    # Ensure package_names is always a list for consistent handling
    if isinstance(package_names, str):
        package_names = [package_names]

    # --- Environment Check (for context and clarity) ---
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "py-mlmodelscope")
    if conda_env:
        print(f"📦 Running in Conda environment: **{conda_env}**")
        if conda_env != "py-mlmodelscope":
            print(
                f"⚠️ Warning: This is not the 'py-mlmodelscope' environment. Proceeding anyway."
            )
    else:
        print(
            "⚠️ Warning: No active Conda environment detected. Installing globally or in the current virtual environment."
        )
    print("-" * 40)

    # Build the full command
    # This correctly passes each package name as a separate argument to pip
    base_command = [sys.executable, "-m", "pip", "install"]
    full_command = base_command + package_names

    print(f"Attempting to install: **{', '.join(package_names)}**")

    try:
        # check_call will raise CalledProcessError if the command fails
        subprocess.check_call(full_command, stderr=subprocess.STDOUT)
        print("\n✅ Installation complete! Your environment is looking spiffy.")
        return 0

    except subprocess.CalledProcessError as e:
        # Catches failures like 'package not found' or build errors
        print("\n❌ Installation Failed!")
        print(f"**Command:** `{' '.join(full_command)}`")
        print(
            f"**Error Details:**\n{e.output.decode()}"
        )  # Decode output for clean error message
        return e.output.decode()

    except FileNotFoundError:
        # Catches if the Python executable itself (sys.executable) is somehow broken/missing
        print(f"\n😱 Critical Error: Python executable not found at {sys.executable}.")
        print("Please ensure your Python and Conda paths are correctly configured.")
        return (
            f"\n😱 Critical Error: Python executable not found at {sys.executable}."
            + "Please ensure your Python and Conda paths are correctly configured."
        )


def extract_pip_modules(text):
    """
    Finds 'pip install [package]' strings and extracts a clean list
    of package names.
    """

    # This regex is still the same: finds 'pip install' and captures
    # the following block of non-space characters.
    matches = re.findall(r"\bpip install\s+([^\s]+)", text)

    cleaned = []
    for pkg in matches:
        # --- THIS IS THE FIX ---
        # Strip a much wider set of common trailing punctuation and quotes
        # that might get accidentally captured by the regex.
        cleaned_pkg = pkg.strip("'\"`).,()")

        # Only add if it's not an empty string after stripping
        if cleaned_pkg:
            cleaned.append(cleaned_pkg)

    return list(set(cleaned))


def run_model_test(model_name, dataset_name_str, test_dir_path):
    """
    Runs the mlmodelscope script for a single model in a conda env and captures the result.
    Saves individual .log and .err files to test_dir_path.
    """
    print(f"--- [STARTING] Test for: {model_name} ---")

    # Base command as a list, now including 'conda run' to use the environment
    command = [
        "conda",
        "run",
        "-n",
        "py-mlmodelscope",  # Specify the conda environment
        "python",
        "run_mlmodelscope.py",
        "--standalone",
        "true",
        "--agent",
        "pytorch",
        "--architecture",
        "gpu",
        "--task",
        "text_to_text",
        "--batch_size",
        "1",
        "--model_name",
        model_name,
        "--dataset_name",
        dataset_name_str,
    ]

    # Join the command list into a string for printing
    print(f"Running command: {' '.join(shlex.quote(arg) for arg in command)}")

    try:
        # Execute the command
        completed_process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,  # We set this to False to handle non-zero exits manually
        )

        # --- NEW: Write individual log files ---
        # Ensure the test directory exists (it should, but good to be safe)
        test_dir_path.mkdir(parents=True, exist_ok=True)
        log_file = test_dir_path / f"{model_name}.log"
        err_file = test_dir_path / f"{model_name}.err"

        try:
            # Write stdout log
            if completed_process.stdout:
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(completed_process.stdout)
                print(f"Saved stdout log to: {log_file}")

            # Write stderr log only if there is error output
            if completed_process.stderr:
                with open(err_file, "w", encoding="utf-8") as f:
                    f.write(completed_process.stderr)
                print(f"Saved stderr log to: {err_file}")

        except IOError as e:
            print(f"--- [WARNING] Could not write log files for {model_name}: {e} ---")
        # --- End of new log writing section ---

        # Check the return code
        if completed_process.returncode == 0:
            print(f"--- [SUCCESS] Test for: {model_name} ---")
            print("Output (last 10 lines):")
            print("\n".join(completed_process.stdout.splitlines()[-10:]))
            return {
                "status": "Success",
                "model": model_name,
                "returncode": completed_process.returncode,
                "output": completed_process.stdout,
                "error": completed_process.stderr,
            }
        else:
            print(f"--- [FAILURE] Test for: {model_name} ---")
            print(f"Return Code: {completed_process.returncode}")
            print("Error output (stderr):")
            print(completed_process.stderr)
            return {
                "status": "Failure",
                "model": model_name,
                "returncode": completed_process.returncode,
                "output": completed_process.stdout,
                "error": completed_process.stderr,
            }

    except FileNotFoundError:
        # This error might now mean 'conda' is not found
        print(f"--- [CRITICAL FAILURE] For: {model_name} ---")
        print("Error: 'conda', 'python' or 'run_mlmodelscope.py' not found.")
        print(
            "Make sure conda is in your PATH and this script is in the same directory as 'run_mlmodelscope.py'."
        )
        return {
            "status": "Critical Failure",
            "model": model_name,
            "error": "Script, Python, or Conda executable not found.",
        }
    except Exception as e:
        print(f"--- [UNEXPECTED ERROR] For: {model_name} ---")
        print(f"An unexpected error occurred: {e}")
        return {"status": "Unexpected Error", "model": model_name, "error": str(e)}


def main():
    # --- Define the Test Directory ---
    TEST_DIR_STR = f"mlmodelscope/pytorch_agent/models/default/text_to_text/test/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    TEST_DIR = Path(TEST_DIR_STR)
    MAX_TRIES_PER_MODEL = 5

    # Create the directory if it doesn't exist
    try:
        TEST_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Using test directory: {TEST_DIR.resolve()}")
    except Exception as e:
        print(f"--- [CRITICAL FAILURE] Could not create directory: {TEST_DIR_STR} ---")
        print(f"Error: {e}")
        return  # Exit if we can't create the log directory

    # --- EDIT THIS LIST ---
    # Add all the model names you want to test here
    models_to_test = [
        "bart_large_emojilm",
        "deepseek_coder_1_3b_instruct",
        "deepseek_r1_0528",
        "deepseek_r1_distill_qwen_1_5b",
        "deepseek_r1_distill_qwen_32b",
        "deepseek_r1_distill_qwen_7b",
        "deepseek_v3_1",
        "dialogpt_medium",
        "dolphin_2_9_1_yi_1_5_34b",
        "glm_4_6_fp8",
        "gpt2",
        "gpt2_large",
        "gpt_oss_120b",
        "gpt_oss_20b",
        "gpt_oss_20b_mxfp4_q8",
        "llama_3_3_70b_instruct_fp8_dynamic",
        "meta_llama_3_1_8b_instruct_bnb_4bit",
        "meta_llama_llama_3_2_3b_instruct_fp16",
        "minimax_m2",
        "mistral_7b_v0_1",
        "olmo_2_0425_1b",
        "phi_2",
        "phi_3_mini_4k_instruct_gptq_4bit",
        "phi_4",
        "qwen2_0_5b",
        "qwen2_5_0_5b",
        "qwen2_5_14b_instruct",
        "qwen2_5_1_5b",
        "qwen2_5_32b_instruct",
        "qwen2_5_32b_instruct_awq",
        "qwen2_5_3b_instruct",
        "qwen2_5_7b",
        "qwen2_5_7b_instruct",
        "qwen2_5_7b_instruct_awq",
        "qwen2_5_coder_7b_instruct_awq",
        "qwen3_0_6b",
        "qwen3_0_6b_base",
        "qwen3_14b",
        "qwen3_1_7b",
        "qwen3_30b_a3b_instruct_2507",
        "qwen3_32b",
        "qwen3_4b",
        "qwen3_4b_instruct_2507",
        "qwen3_8b",
        "qwen3_next_80b_a3b_instruct",
        "schematron_3b",
        "smollm2_135m",
        "stablebeluga2",
        "tiny_qwen2forcausallm_2_5",
        "tinyllama_1_1b_chat_v1_0",
    ]
    # --- END EDIT ---

    # This is the --dataset_name argument from your command
    dataset_name_str = '[{"src":"What is earth?","inputType":"TEXT"}]'

    all_results = {}
    success_count = 0
    failure_count = 0
    models_need_more_GPU = []

    for model in models_to_test:
        tries = 0
        while tries < MAX_TRIES_PER_MODEL:
            if tries != 0:
                print(f"Going for try {tries+1}")
            start_time = time.time()
            readable_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = run_model_test(
                model_name=model,
                dataset_name_str=dataset_name_str,
                test_dir_path=TEST_DIR,
            )
            end_time = time.time()
            duration = round(end_time - start_time, 2)
            result["startTime"] = readable_start
            result["runTime"] = duration
            if "CUDA out of memory" in result["error"]:
                models_need_more_GPU.append(model)
                break
            elif "pip install" in result["error"]:
                models_to_install = extract_pip_modules(result["error"])
                print("Trying to install", models_to_install)
                if pip_error := install_packages_in_conda(models_to_install):
                    result["error"] += "\n" + pip_error
                    break

            else:
                break
            tries += 1
        all_results.setdefault(model, []).append(result)
        if result["status"] == "Success":
            success_count += 1
        else:
            failure_count += 1
        print(f"--- [FINISHED] Test for: {model} ---\n")

    # Save full summary JSON results to the TEST_DIR
    results_filename = TEST_DIR / f"model_test_results.json"
    bigger_than_GPU_File = TEST_DIR / f"GPU_overload_models.txt"
    try:
        with open(results_filename, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Full summary results saved to '{results_filename}'")
    except IOError as e:
        print(f"Error saving summary JSON to {results_filename}: {e}")
    try:
        with open(bigger_than_GPU_File, "w") as f:
            f.write("\n".join(models_need_more_GPU))
        print(f"Full summary results saved to '{bigger_than_GPU_File}'")
    except IOError as e:
        print(f"Error saving GPU short on memory to {bigger_than_GPU_File}: {e}")

    # Print final summary
    print("=" * 40)
    print("           Test Run Summary")
    print("=" * 40)
    print(f"Total Tests: {len(models_to_test)}")
    print(f"Successful:  {success_count}")
    print(f"Failed:      {failure_count}")
    print(f"\nLogs and summary JSON saved in: {TEST_DIR.resolve()}")
    print("\nDetailed Summary:")
    for model_name, results_list in all_results.items():
        # Get the final result from the list (it's the last one)
        final_result = results_list[-1]
        print(f"  - {final_result['model']}: {final_result['status']}")
        if final_result["status"] != "Success":
            error_preview = final_result.get("error", "No error message.").splitlines()
            error_snippet = f": {error_preview[0]}" if error_preview else ""
            print(
                f"    - Details: Return Code {final_result.get('returncode', 'N/A')}{error_snippet}"
            )


if __name__ == "__main__":
    main()
