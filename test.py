import re, subprocess, sys, os, shutil, json, time, shlex
from pathlib import Path
from datetime import datetime

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


def extract_missing_module_name(error_message):
    """
    Finds the missing module name from a ModuleNotFoundError string.

    Example input: "ModuleNotFoundError: No module named 'timm'"
    Example output: "timm"
    """

    # This regex looks for the literal string "No module named '"
    # and then captures everything inside the following single quotes.
    match = re.search(r"No module named '([^']+)'", error_message)

    if match:
        # match.group(0) is the whole matched string: "No module named 'timm'"
        # match.group(1) is *only* the part in the parentheses: "timm"
        return match.group(1)

    return None


def run_model_test(model_name, dataset_name_str, test_dir_path, task):
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
        task,
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
        # # test_dir_path.mkdir(parents=True, exist_ok=True)
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


def main(task, dataset_name_src, models_to_test):
    # --- Define the Test Directory ---
    TEST_DIR_STR = f"mlmodelscope/pytorch_agent/models/default/{task}/test/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
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
    # --- END EDIT ---

    # This is the --dataset_name argument from your command

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
            try:
                result = run_model_test(
                    model_name=model,
                    dataset_name_str=dataset_name_src,
                    test_dir_path=TEST_DIR,
                    task=task,
                )
                end_time = time.time()
                duration = round(end_time - start_time, 2)
                result["startTime"] = readable_start
                result["runTime"] = duration

            except Exception as e:
                e_str = str(e)

                if "CUDA out of memory" in e_str:
                    models_need_more_GPU.append(model)
                    break
                
                elif "pip install" in e_str:
                    models_to_install = extract_pip_modules(e_str)
                    print("Trying to install", models_to_install)

                    pip_error = install_packages_in_conda(models_to_install)
                    if pip_error:
                        e = Exception(e_str + "\n" + pip_error)
                        break
                    
                elif "ModuleNotFoundError" in e_str:
                    missing_module_match = extract_missing_module_name(e_str)
                    if missing_module_match:
                        print(f"Trying to install missing module: {missing_module_match}")
                        pip_error = install_packages_in_conda(missing_module_match)
                        if pip_error:
                            e = Exception(e_str + "\n" + pip_error)
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
        cache_dir = os.path.expanduser("~/.cache/huggingface/")
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"--- Deleted ~/.cache/huggingface/ ---\n")

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
    dirs = [
        "mlmodelscope/pytorch_agent/models/default/automatic_speech_recognition/wav2vec2_large_xlsr_punjabi/model.py",
        "mlmodelscope/pytorch_agent/models/default/depth_estimation/coreml_sam2_1_small/model.py",
        "mlmodelscope/pytorch_agent/models/default/document_visual_question_answering/layoutlm_document_qa/model.py",
        "mlmodelscope/pytorch_agent/models/default/image_captioning/pix2struct_tiny_random/model.py",
        "mlmodelscope/pytorch_agent/models/default/image_editing/yoso_normal_v1_8_1/model.py",
        "mlmodelscope/pytorch_agent/models/default/image_object_detection/dfine_small_coco/model.py",
        "mlmodelscope/pytorch_agent/models/default/image_semantic_segmentation/seg_zero_7b/model.py",
        "mlmodelscope/pytorch_agent/models/default/music_generation/qwen2_5_omni_wo_video_1016/model.py",
        "mlmodelscope/pytorch_agent/models/default/sentiment_analysis/deberta_v3_xsmall_mnli_fever_anli_ling_binary/model.py",
        "mlmodelscope/pytorch_agent/models/default/text_to_image/stablematerials/model.py",
        "mlmodelscope/pytorch_agent/models/default/text_to_text/t5_small_booksum/model.py",
        "mlmodelscope/pytorch_agent/models/default/video_classification/xclip_base_patch16/model.py",
        "mlmodelscope/pytorch_agent/models/default/visual_question_answering/vl_rethinker_72b/model.py",
        "mlmodelscope/pytorch_agent/models/default/text_to_image/testllm/model.py",
        "mlmodelscope/pytorch_agent/models/default/image_classification/siglip2_base_patch16_224/model.py",
        "mlmodelscope/pytorch_agent/models/default/text_to_image/testllm/model.py",
        "mlmodelscope/pytorch_agent/models/default/music_generation/speecht5_finetuned_voxpopuli_es/model.py",
    ]
    myJson = dict(json.loads(open("./test.json").read()))
    for modality in myJson:
        dir = myJson[modality]["dir"]
        test = json.dumps(myJson[modality]["test"])
        modelName = [x.split("/")[-2] for x in dirs if dir in x][0]
        task = [x.split("/")[-3] for x in dirs if dir in x][0]
        print(task,modelName)
        run_model_test(modelName,test,Path("./test_run/"), task )
