"""
LangChain automation to generate input configurations for existing models.

This script:
1. Iterates over modalities and models from automateAll.py structure
2. Checks if models exist in the filesystem
3. Fetches Hugging Face documentation for existing models
4. Uses Gemini to determine input types and counts
5. Generates JSON configuration similar to categorized_huggingFaceModels.json
"""

import os
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, Optional
from datetime import datetime


class InputConfig(BaseModel):
    """Configuration for model input types and counts."""

    inputs: Dict[str, int] = Field(
        description="Dictionary mapping input type names to their counts. "
        "Positive integers indicate exact count (e.g., 1 means exactly 1). "
        "-1 indicates variable/any number of that input type. "
        "Valid input types: 'Image', 'Text', 'Audio', 'Video', 'Tensor', etc. "
        "Example: {'Image': 1, 'Text': 1} for models requiring exactly 1 image and 1 text, "
        "or {'Image': -1} for models accepting any number of images."
    )


def fetch_huggingface_doc(model_name: str) -> Optional[str]:
    """Fetch documentation from Hugging Face model page."""
    try:
        url = f"https://huggingface.co/{model_name}"
        print(f"Fetching documentation from {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        main_content = (
            soup.find("model-card-content") or soup.find("main") or soup.find("body")
        )

        if not main_content:
            print(f"Could not find main content on page for {model_name}")
            return None

        login_link = main_content.find(
            "a", href=lambda h: h and h.startswith("/login?next=")
        )
        if login_link:
            print(f"Login required for {model_name}, skipping.")
            return None

        context_text = main_content.get_text(separator=" ", strip=True)
        max_length = 15000
        if len(context_text) > max_length:
            context_text = context_text[:max_length] + "\n... (content truncated)"

        print(f"Successfully fetched documentation for {model_name}")
        return context_text

    except Exception as e:
        print(f"Error fetching documentation for {model_name}: {e}")
        return None


def check_model_exists(modality_dir: str, model_name: str) -> bool:
    """Check if model exists in the filesystem."""
    model_folder_name = (
        model_name.split("/")[-1].replace("-", "_").replace(".", "_").lower()
    )
    model_py_path = os.path.join(
        "mlmodelscope/pytorch_agent/models/default",
        modality_dir,
        model_folder_name,
        "model.py",
    )
    return os.path.exists(model_py_path)


def main():
    # Load configuration files
    print("Loading configuration files...")
    with open("test.json", "r") as f:
        modality_mapping = json.load(f)

    with open("categorized_huggingFaceModels.json", "r") as f:
        categorized_models = json.load(f)

    # Setup LLM
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY not found in .env file or environment.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        model_kwargs={"thinkingBudget": -1},
        temperature=0,
        convert_system_message_to_human=True,
    )

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert in machine learning model input requirements. Your task is to analyze a Hugging Face model's documentation and determine what inputs the model requires.

You will be given:
1. A model identifier (e.g., "google/vit-base-patch16-224")
2. The documentation/content from the model's Hugging Face page

Your task is to determine:
- What types of inputs the model accepts (Image, Text, Audio, Video, Tensor, etc.)
- How many of each input type are required

**IMPORTANT RULES:**
1. **Input Types**: Use standard names like "Image", "Text", "Audio", "Video", "Tensor"
2. **Counts**: 
   - Use positive integers (1, 2, 3, etc.) for exact counts
   - Use -1 for variable/any number of that input type (e.g., batch processing)
3. **Common Patterns**:
   - Image Classification: {{"Image": 1}} (exactly 1 image)
   - Text Classification: {{"Text": 1}} (exactly 1 text)
   - Text-to-Image: {{"Text": 1}} (exactly 1 text prompt)
   - Image-to-Text: {{"Image": 1}} (exactly 1 image)
   - Visual Question Answering: {{"Image": 1, "Text": 1}} (1 image + 1 question text)
   - Object Detection: {{"Image": 1}} (exactly 1 image)
   - Text-to-Text: {{"Text": 1}} (exactly 1 text)
   - Batch processing (multiple inputs): {{"Image": -1}} or {{"Text": -1}} (any number)
   - Multi-modal: {{"Image": 1, "Text": 1}} (both required)

4. **Analyze the model's purpose**:
   - Check the model card for input examples
   - Look for preprocessing steps (image preprocessing, tokenization, etc.)
   - Check if the model accepts batches (then use -1)
   - Check if it's multi-modal (multiple input types)

**Examples:**

Example 1: Image Classification Model
Model: "google/vit-base-patch16-224"
Inputs: {{"Image": 1}}
Explanation: Takes exactly one image as input.

Example 2: Text-to-Image Generation
Model: "CompVis/stable-diffusion-v1-4"
Inputs: {{"Text": 1}}
Explanation: Takes exactly one text prompt as input.

Example 3: Visual Question Answering
Model: "Salesforce/blip-vqa-base"
Inputs: {{"Image": 1, "Text": 1}}
Explanation: Takes exactly one image and one question text.

Example 4: Text Classification (batch processing)
Model: "distilbert-base-uncased-finetuned-sst-2-english"
Inputs: {{"Text": -1}}
Explanation: Can process any number of texts (batch processing).

Example 5: Image-to-Text (Captioning)
Model: "Salesforce/blip-image-captioning-base"
Inputs: {{"Image": 1}}
Explanation: Takes exactly one image as input.

Example 6: Text-to-Text Generation
Model: "gpt2"
Inputs: {{"Text": 1}}
Explanation: Takes exactly one text as input.

Respond ONLY with the JSON structure matching the InputConfig schema:
{{"inputs": {{"InputType": count, ...}}}}
""",
            ),
            (
                "human",
                """
Model Identifier: '{model_identifier}'

Model Documentation:
---
{model_page_context}
---

Based on the documentation above, determine the input types and counts required by this model.
Return a JSON object with the "inputs" field containing a dictionary of input types to counts.
""",
            ),
        ]
    )

    parser = JsonOutputParser(pydantic_object=InputConfig)
    chain = prompt | llm | parser

    # Generate input configs for all existing models
    failed_models = []  # List of tuples: (modality_name, model_name, error_reason)
    skipped_models = []  # List of tuples: (modality_name, model_name)
    processed_modalities = 0
    total_models_processed = 0

    for modality_name, modality_info in modality_mapping.items():
        if modality_name not in categorized_models:
            print(
                f"Modality '{modality_name}' not found in categorized_models.json, skipping."
            )
            continue

        modality_dir = modality_info.get("dir")
        if not modality_dir:
            print(f"No directory mapping for modality '{modality_name}', skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Processing modality: {modality_name} (dir: {modality_dir})")
        print(f"{'='*60}")

        # Path to the input config JSON file for this modality
        modality_base_dir = os.path.join(
            "mlmodelscope/pytorch_agent/models/default", modality_dir
        )
        input_config_file = os.path.join(modality_base_dir, "input_configs.json")

        # Load existing config if it exists
        existing_config = {}
        if os.path.exists(input_config_file):
            try:
                with open(input_config_file, "r") as f:
                    existing_config = json.load(f)
                print(f"Loaded existing config with {len(existing_config)} models")
            except Exception as e:
                print(f"Warning: Could not load existing config: {e}")
                existing_config = {}

        models = categorized_models[modality_name].get("allModels", [])
        modality_config = existing_config.copy()  # Start with existing config
        models_processed_this_modality = 0

        for model_name in models:
            # Check if model exists
            if not check_model_exists(modality_dir, model_name):
                print(f"Model '{model_name}' does not exist, skipping.")
                skipped_models.append((modality_name, model_name))
                continue

            # Update existing configs if they exist (merge/update behavior)
            if model_name in modality_config:
                print(f"Model '{model_name}' already has config, will update it.")

            print(f"\n--- Processing: {model_name} ---")

            # Fetch Hugging Face documentation
            model_doc = fetch_huggingface_doc(model_name)
            if not model_doc:
                error_reason = "Could not fetch documentation from Hugging Face"
                print(f"Could not fetch documentation for {model_name}, skipping.")
                failed_models.append((modality_name, model_name, error_reason))
                continue

            # Generate input config using LLM
            try:
                print(f"Invoking LLM to generate input config for {model_name}...")
                result = chain.invoke(
                    {
                        "model_identifier": model_name,
                        "model_page_context": model_doc,
                    }
                )
                # Extract the inputs dictionary from the result
                input_config = result.get("inputs", {})
                if input_config and isinstance(input_config, dict):
                    # Validate that all values are integers
                    valid_config = {}
                    for key, value in input_config.items():
                        if isinstance(value, int):
                            valid_config[key] = value
                        else:
                            print(
                                f"Warning: Invalid count type for {key} in {model_name}: {value}, skipping."
                            )

                    if valid_config:
                        modality_config[model_name] = valid_config
                        models_processed_this_modality += 1
                        total_models_processed += 1
                        print(
                            f"✓ Successfully generated config for {model_name}: {valid_config}"
                        )
                    else:
                        error_reason = (
                            "No valid input config generated (invalid or empty config)"
                        )
                        print(f"✗ No valid input config generated for {model_name}")
                        failed_models.append((modality_name, model_name, error_reason))
                else:
                    error_reason = "No input config returned from LLM"
                    print(f"✗ No input config generated for {model_name}")
                    failed_models.append((modality_name, model_name, error_reason))
            except Exception as e:
                error_reason = f"Exception: {str(e)}"
                print(f"✗ Error generating input config for {model_name}: {e}")
                import traceback

                traceback.print_exc()
                failed_models.append((modality_name, model_name, error_reason))

        # Save config for this modality (only if we have configs)
        if modality_config:
            # Ensure directory exists
            os.makedirs(modality_base_dir, exist_ok=True)

            # Save updated config
            with open(input_config_file, "w") as f:
                json.dump(modality_config, f, indent=2)

            print(
                f"\n✓ Processed {models_processed_this_modality} new models for {modality_name}"
            )
            print(f"✓ Total models in config: {len(modality_config)}")
            print(f"✓ Saved config to: {input_config_file}")
            processed_modalities += 1
        else:
            print(f"\n✗ No models with configs for {modality_name}")

    # Save common log file for failed models
    log_file = "model_input_configs_failed.log"
    if failed_models:
        # Append to log file if it exists, otherwise create new one
        file_exists = os.path.exists(log_file)
        with open(log_file, "a") as f:
            if not file_exists:
                f.write("# Failed Models Log\n")
                f.write("# Format: Timestamp | Modality | Model Name | Error Reason\n")
                f.write("# " + "=" * 80 + "\n\n")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n# Run at {timestamp}\n")
            f.write("# " + "-" * 80 + "\n")
            for modality, model, reason in failed_models:
                f.write(f"{timestamp} | {modality} | {model} | {reason}\n")
        print(
            f"\nFailed models log saved to: {log_file} ({len(failed_models)} entries)"
        )

    # Save common log file for skipped models (optional - only if user wants to track these)
    skipped_log_file = "model_input_configs_skipped.log"
    if skipped_models:
        file_exists = os.path.exists(skipped_log_file)
        with open(skipped_log_file, "a") as f:
            if not file_exists:
                f.write("# Skipped Models Log (not found in filesystem)\n")
                f.write("# Format: Timestamp | Modality | Model Name\n")
                f.write("# " + "=" * 80 + "\n\n")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n# Run at {timestamp}\n")
            f.write("# " + "-" * 80 + "\n")
            for modality, model in skipped_models:
                f.write(f"{timestamp} | {modality} | {model}\n")
        print(
            f"Skipped models log saved to: {skipped_log_file} ({len(skipped_models)} entries)"
        )

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"{'='*60}")
    print(f"Total modalities processed: {processed_modalities}")
    print(f"Total new models with configs: {total_models_processed}")
    print(f"Failed models: {len(failed_models)}")
    if failed_models:
        print(f"  Failed models log: {log_file}")
    print(f"Skipped models (not found): {len(skipped_models)}")
    if skipped_models:
        print(f"  Skipped models log: {skipped_log_file}")

    if failed_models:
        print(f"\nFailed models (showing first 10):")
        for modality, model, reason in failed_models[:10]:
            print(f"  - [{modality}] {model}: {reason}")
        if len(failed_models) > 10:
            print(
                f"  ... and {len(failed_models) - 10} more (see log file for full list)"
            )


if __name__ == "__main__":
    main()
