import re, subprocess, sys, os, shutil, json, time, shlex
from pathlib import Path
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

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


def debug_with_gemini(file_name: str, error_message: str) -> int:
    """
    Attempts to debug a Python file using the Gemini API via LangChain.
    Validates the fix with a syntax check before overwriting.

    Args:
        file_name: The path to the Python file with the bug.
        error_message: The error message received when running the file.

    Returns:
        0 if the file was successfully fixed and updated.
        -1 if any step failed (file read, API error, no fix, syntax check, file write).
    """
    
    # 1. Check for Google API Key
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY environment variable not set.", file=sys.stderr)
        print("Please set this environment variable with your API key.", file=sys.stderr)
        return -1

    # 2. Read the original code from the file
    try:
        with open(file_name, 'r') as f:
            original_code = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_name}", file=sys.stderr)
        return -1
    except IOError as e:
        print(f"Error reading file {file_name}: {e}", file=sys.stderr)
        return -1
    except Exception as e:
        print(f"An unexpected error occurred during file read: {e}", file=sys.stderr)
        return -1

    # 3. Set up LangChain components
    try:
        # Initialize the Google Generative AI model
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

        # Define the system prompt to instruct the model
        system_template = """
You are an expert Python debugger. Your task is to fix the provided Python code based on the given error message.

Rules:
1.  Analyze the code and the error message.
2.  If you can fix the code, respond *only* with the complete, corrected Python code, inside a single markdown code block (```python ... ```).
3.  Do not include *any* explanation, greeting, or text before or after the code block. Your response must contain *only* the code block.
4.  If you *cannot* fix the code, or if the error message is insufficient, respond *only* with the exact string: CANNOT_FIX
"""
        
        # Define the human prompt that will contain the code and error
        human_template = """
Here is the code that needs debugging:
---CODE---
{code}
---END CODE---

Here is the error message I received:
---ERROR---
{error}
---END ERROR---
"""
        # Create the full prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        # Define a simple string output parser
        output_parser = StrOutputParser()
        
        # Create the chain using LangChain Expression Language (LCEL)
        chain = prompt | llm | output_parser

    except Exception as e:
        print(f"Error setting up LangChain components: {e}", file=sys.stderr)
        return -1

    # 4. Run the chain (invoke the model)
    try:
        print(f"Sending code from {file_name} to Gemini for debugging...")
        response = chain.invoke({
            "code": original_code,
            "error": error_message
        })
        
        # 5. Process the response
        response_trimmed = response.strip()
        
        if response_trimmed == "CANNOT_FIX":
            print("Gemini reported it cannot fix the code.")
            return -1

        # 6. Extract code from the markdown block
        # Use re.DOTALL (s) flag to make '.' match newlines
        match = re.search(r"```python\n(.*?)\n```", response_trimmed, re.DOTALL | re.IGNORECASE)
        
        fixed_code = ""
        if match:
            fixed_code = match.group(1).strip()
        else:
            # Fallback: Check if the model *only* returned code without the block
            if "def " in response_trimmed or "import " in response_trimmed or "print(" in response_trimmed:
                 print("Warning: Model returned code without markdown block. Trying to use it anyway.")
                 fixed_code = response_trimmed
            else:
                print("Error: Model response was not in the expected format (no ```python block).", file=sys.stderr)
                print("---GEMINI RESPONSE---")
                print(response)
                print("---END RESPONSE---")
                return -1

        if not fixed_code:
             print("Error: Extracted fixed code is empty.", file=sys.stderr)
             return -1

        # 7. NEW: Validate syntax and write the fixed code
        
        # Define the syntax check function
        # os.system returns 0 on success
        check_syntax = lambda fn: os.system(f"{sys.executable} -m py_compile {fn}")
        
        temp_dir_name = f"_gemini_debug_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        temp_file_name = os.path.join(temp_dir_name, "fixed_code.py")
        
        try:
            # Create a unique temporary directory in the current folder
            os.makedirs(temp_dir_name, exist_ok=True)

            # Write the fixed code to a file inside the temp directory
            with open(temp_file_name, 'w') as temp_file:
                temp_file.write(fixed_code)

            # Run the syntax check on the temporary file
            print(f"Checking syntax of Gemini's fix in {temp_file_name}...")
            exit_code = check_syntax(temp_file_name)

            if exit_code != 0:
                print(f"Error: Gemini's fix failed the syntax check (exit code: {exit_code}).", file=sys.stderr)
                print("Original file was NOT modified.")
                return -1
            
            print("Syntax check passed.")
            
            # 8. Write the validated fixed code back to the original file
            try:
                with open(file_name, 'w') as f:
                    f.write(fixed_code)
                print(f"Successfully fixed and updated {file_name}.")
                return 0  # Success!
            except IOError as e:
                print(f"Error writing fixed code to {file_name}: {e}", file=sys.stderr)
                return -1

        except Exception as e:
            print(f"An error occurred during validation/writing: {e}", file=sys.stderr)
            return -1
        finally:
            # Clean up the temporary file
            if temp_file_name and os.path.exists(temp_file_name):
                os.remove(temp_file_name)
                print(f"Cleaned up temporary file: {temp_file_name}")
            if os.path.isdir(temp_dir_name):
                shutil.rmtree(temp_dir_name)
                print(f"Deleted directory: {temp_dir_name}")

    except Exception as e:
        # This catches API errors, rate limits, etc.
        print(f"An error occurred while communicating with Gemini: {e}", file=sys.stderr)
        return -1


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
        return 1  # Exit if we can't create the log directory, return failure

    # --- EDIT THIS LIST ---
    # Add all the model names you want to test here
    # --- END EDIT ---

    # This is the --dataset_name argument from your command

    all_results = {}
    success_count = 0
    failure_count = 0
    models_need_more_GPU = []

    for model in models_to_test:
        MODEL_FILE = f"mlmodelscope/pytorch_agent/models/default/{task}/{model}/model.py"
        tries = 0
        result = {}  # Initialize result here
        while tries < MAX_TRIES_PER_MODEL:
            if tries != 0:
                print(f"Going for try {tries+1}")
            start_time = time.time()
            readable_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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

            # --- FIXED LOGIC ---
            # Check the result status
            if result["status"] == "Success":
                break  # Successful run, exit the while loop

            # If it failed, check why. Combine output and error for searching.
            error_str = result.get("error", "") + result.get("output", "")

            if "CUDA out of memory" in error_str:
                print(
                    "Detected CUDA out of memory. Adding to list and stopping tries."
                )
                models_need_more_GPU.append(model)
                break  

            elif "pip install" in error_str:
                print("Detected 'pip install' recommendation.")
                models_to_install = extract_pip_modules(error_str)
                if models_to_install:
                    print(f"Trying to install: {models_to_install}")
                    pip_error = install_packages_in_conda(models_to_install)
                    if pip_error:
                        print("Installation failed. Stopping tries.")
                        result["error"] = (
                            result.get("error", "") + "\n" + pip_error
                        )
                        break  
            elif "ModuleNotFoundError" in error_str:
                print("Detected 'ModuleNotFoundError'.")
                missing_module = extract_missing_module_name(error_str)
                if missing_module:
                    print(
                        f"Trying to install missing module: {missing_module}"
                    )
                    pip_error = install_packages_in_conda(missing_module)
                    if pip_error:
                        print("Installation failed. Stopping tries.")
                        result["error"] = (
                            result.get("error", "") + "\n" + pip_error
                        )
                        break
                else:
                    print(
                        "Found 'ModuleNotFoundError' but couldn't parse module. Stopping tries."
                    )
                    break  # Avoid infinite loop

            else:
                debug_check = debug_with_gemini(file_name= MODEL_FILE, error_message= error_str)
                if debug_check!=0:
                    break
            tries += 1

        all_results.setdefault(model, []).append(result)
        
        if result.get("status") == "Success":
            success_count += 1
        else:
            failure_count += 1
            
        print(f"--- [FINISHED] Test for: {model} ---\n")
        cache_dir = os.path.expanduser("~/.cache/huggingface/")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"--- Deleted ~/.cache/huggingface/ ---\n")
        else:
            print(f"--- Cache dir not found. Skipping delete: ~/.cache/huggingface/ ---\n")

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
        print(f"  - {final_result.get('model', model_name)}: {final_result.get('status', 'Unknown')}")
        if final_result.get("status") != "Success":
            error_preview = final_result.get("error", "No error message.").splitlines()
            error_snippet = f": {error_preview[0]}" if error_preview else ""
            print(
                f"    - Details: Return Code {final_result.get('returncode', 'N/A')}{error_snippet}"
            )
            
    # Return the number of failures for sys.exit()
    return failure_count


if __name__ == "__main__":
    load_dotenv()  # Load variables from .env file
    overall_failures = 0
    try:
        dirs = [
            "mlmodelscope/pytorch_agent/models/default/automatic_speech_recognition/wav2vec2_large_xlsr_punjabi/model.py",
            "mlmodelscope/pytorch_agent/models/default/depth_estimation/coreml_sam2_1_small/model.py",
            "mlmodelscope/pytorch_agent/models/default/document_visual_question_answering/layoutlm_document_qa/model.py",
            "mlmodelscope/pytorch_agent/models/default/image_captioning/pix2struct_tiny_random/model.py",
            "mlmodelscope/pytorch_agent/models/default/image_classification/nsfw_image_detector/model.py",
            "mlmodelscope/pytorch_agent/models/default/image_editing/yoso_normal_v1_8_1/model.py",
            "mlmodelscope/pytorch_agent/models/default/image_object_detection/dfine_small_coco/model.py",
            "mlmodelscope/pytorch_agent/models/default/image_semantic_segmentation/seg_zero_7b/model.py",
            "mlmodelscope/pytorch_agent/models/default/music_generation/qwen2_5_omni_wo_video_1016/model.py",
            "mlmodelscope/pytorch_agent/models/default/sentiment_analysis/deberta_v3_xsmall_mnli_fever_anli_ling_binary/model.py",
            "mlmodelscope/pytorch_agent/models/default/text_to_image/stablematerials/model.py",
            "mlmodelscope/pytorch_agent/models/default/text_to_text/t5_small_booksum/model.py",
            "mlmodelscope/pytorch_agent/models/default/video_classification/xclip_base_patch16/model.py",
            "mlmodelscope/pytorch_agent/models/default/visual_question_answering/vl_rethinker_72b/model.py"
        ]
        
        json_file_path = "./test.json"
        if not os.path.exists(json_file_path):
            print(f"❌ Error: JSON file not found at {json_file_path}")
            sys.exit(1) # Exit with error
            
        with open(json_file_path, "r") as f:
            myJson = dict(json.loads(f.read()))

        for modality in myJson:
            dir_key = myJson[modality].get("dir")
            test_data = myJson[modality].get("test")
            
            if not dir_key or test_data is None:
                print(f"⚠️ Skipping modality '{modality}': missing 'dir' or 'test' key.")
                continue

            test = json.dumps(test_data)

            # Find model and task, with error checking
            modelName_matches = [x.split("/")[-2] for x in dirs if dir_key in x]
            task_matches = [x.split("/")[-3] for x in dirs if dir_key in x]
            
            if not modelName_matches or not task_matches:
                print(f"⚠️ Skipping modality '{modality}': could not find matching model/task for dir '{dir_key}'")
                overall_failures += 1
                continue
                
            modelName = modelName_matches[0]
            task = task_matches[0]
            
            print(f"\n{'='*20} 🚀 Starting Test for Task: {task}, Model: {modelName} {'='*20}\n")
            # main() now returns the number of failures for that run
            failures = main(task=task, dataset_name_src=test, models_to_test=[modelName])
            overall_failures += failures
            print(f"\n{'='*20} 🏁 Finished Test for Task: {task}, Model: {modelName} {'='*20}\n")
            
    except json.JSONDecodeError:
        print("❌ Error: Failed to decode './test.json'. Please check for syntax errors.")
        overall_failures += 1
    except FileNotFoundError:
        print("❌ Error: './test.json' file not found.")
        overall_failures += 1
    except Exception as e:
        print(f"❌ An unexpected error occurred in the main execution block: {e}")
        overall_failures += 1
        
    finally:
        # --- This is the new exit logic ---
        if overall_failures > 0:
            print(f"\n🚫 Script finished with {overall_failures} total failures.")
            sys.exit(1) # Exit with a non-zero status code
        else:
            print("\n✅ All tests passed. Script finished successfully.")
            sys.exit(0) # Exit with status code 0