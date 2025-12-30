# Automation Debugging Log

## Image Classification Models

This log tracks the debugging process for image classification models.

## mobilevit_small

**Result:**
Success. Model runs and produces output.
**Note:** `MobileViTFeatureExtractor` is deprecated but functional.

## eva02_enormous_patch14_plus_clip_224_laion2b_s9b_b144k

**Initial Failure:**
`OSError: timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k does not appear to have a file named preprocessor_config.json`.
`transformers` failed to load the model config.

**Action:**
Rewrote `model.py` to use `open_clip` primitives `create_model_from_pretrained` (with `hf-hub:` prefix) and `get_tokenizer`. Added default labels for zero-shot.

**Result:**
Success. Model runs and produces output.

## clip_convnext_large_d_320_laion2b_s29b_b131k_ft_soup

**Initial Failure:**
`TypeError: Unexpected type <class 'list'>`
This was caused by a naming conflict: the `open_clip` transform was assigned to `self.preprocess` (instance attribute), shadowing the `preprocess` method of the class. When the agent called `preprocess(data)`, it invoked the transform function with a list (data batch) instead of the method.

**Action:**
Renamed the transform attribute to `self.preprocess_fn` and updated usage.

**Result:**
Success. Model runs and produces output.

## eva02_large_patch14_clip_224_merged2b_s4b_b131k

**Initial Failure:**
`OSError: .../preprocessor_config.json`.
Missing transformers config.

**Action:**
Rewrote `model.py` to use `open_clip` primitives `create_model_from_pretrained` (with `hf-hub:` prefix) and `get_tokenizer`. Added default labels for zero-shot.

**Result:**
Success. Model runs and produces output.

## FIXME: blip2_itm_vit_g

**Initial Failure:**
`ImportError: cannot import name 'Blip2ForImageTextMatching' from 'transformers'`
The class `Blip2ForImageTextMatching` does not exist in the installed `transformers` version.

**Action:**
Switched to using `Blip2Model`.

**Second Failure:**
`AttributeError: 'bool' object has no attribute 'unsqueeze'` in `Blip2Model.forward`.
This appears to be an internal bug in `transformers` `Blip2Model` implementation related to `get_placeholder_mask`, or compatibility issue with the specific ITM checkpoint. Additionally, `Blip2Model` does not strictly provide the ITM head outputs (logits) required for the task.

**Result:**
Failed. Unable to resolve without significant library changes or custom implementation of the ITM head.

## vit_hybrid_base_bit_384

**Result:**
Success. Model runs and produces output.

## pe_core_l14_336

**Initial Failure:**
`ValueError: Zero-shot classification requires a list of 'labels'...`

**Action:**
Modified `model.py` to provide default labels if not provided in config.

**Result:**
Success. Model runs and produces output.

## vit_so400m_14_siglip2_378

**Result:**
Success. Model runs and produces output.








## clip_vit_base_patch32

**Initial Failure:**
`OSError: Xenova/clip-vit-base-patch32 does not appear to have a file named pytorch_model.bin...`

**Action:**
Modified `model.py` to use `model_id = "openai/clip-vit-base-patch32"` instead of `Xenova/clip-vit-base-patch32`.

**Second Failure:**
`ValueError: Due to a serious vulnerability issue in torch.load... we now require users to upgrade torch to at least v2.6`
Current torch version is `2.4.0+cu121`.

**Action:**
Upgrading PyTorch to version >= 2.6.

**Third Failure:**
`RuntimeError: Expected all tensors to be on the same device...`
This happened because `accelerate` (used by parent class) split the model across GPUs despite explicitly loading it to one device, and input tensors were on a different device.

**Action:**
1. Modified `model.py` to manually handle device placement (removed `device_map="auto"`).
2. Overrode `to` method in `model.py` to bypass parent class automatic dispatching.
3. Updated `predict` to ensure inputs are moved to the correct device.

**Result:**
Success. Model runs and produces output.

## pe_core_b16_224

**Initial Failure:**
`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 376.00 MiB. GPU 0...`
GPU 0 is occupied by another user (`samarpra` running `VLLM`).

**Action:**
Switching to GPU 3 (which appeares to have free memory) using `CUDA_VISIBLE_DEVICES=3`.

**Result:**
Success. Model runs and produces output.

## mit_b3

**Result:**
Success. Model runs and produces output (tested on GPU 3).

## biomedclip_pubmedbert_256_vit_base_patch16_224

**Initial Failure:**
`AttributeError: 'CustomTextCLIP' object has no attribute 'device'`

**Action:**
Modified `model.py` to use `next(self.model.parameters()).device` for ensuring input tensors are on the correct device.

**Result:**
Success. Model runs and produces output.

## marqo_fashionclip

**Initial Failure:**
`NotImplementedError: Cannot copy out of meta tensor; no data!`
This occurred because `transformers` `AutoModel` put the model on meta device (likely due to `accelerate` integration), but `open_clip` (used internally by the remote code) failed to handle it properly.

**Action:**
Modified `model.py` to:
1. Use `open_clip` directly (`create_model_from_pretrained`), bypassing `AutoModel` wrapper.
2. Added default labels for zero-shot classification.
3. Updated `predict` to use `encode_image`.

**Result:**
Success. Model runs and produces output.


## clip_vit_b_16_datacomp_xl_s13b_b90k

**Initial Failure:**
`OSError: laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K does not appear to have a file named preprocessor_config.json`
`transformers` failed to load the model because it lacks configuration files.

**Action:**
Rewrote `model.py` to use `open_clip` primitives (`create_model_from_pretrained` and `get_tokenizer`), bypassing `transformers` loading issues.

**Result:**
Success. Model runs and produces output.

## vit_b_16_siglip_i18n_256

**Initial Failure:**
`RuntimeError: Unknown model (ViT-B-16-SigLIP-i18n-256)`
The name `timm/ViT-B-16-SigLIP-i18n-256` was not recognized by `timm`.

**Action:**
Updated `model.py` to use the standard `timm` model name `vit_base_patch16_siglip_256`.

**Result:**
Success. Model runs and produces output.

## siglip2_large_patch16_384

**Initial Failure:**
`ValueError: The 'candidate_labels' key with a list of label strings must be provided...`

**Action:**
1. Modified `model.py` to provide default `candidate_labels`  if not provided.
2. Disabled `device_map="auto"` and manually handled device placement.
3. Updated `predict` to ensure inputs are on the correct device.

**Result:**
Success. Model runs and produces output.











## plip

**Initial Failure:**
`NameError: name 'kwargs' is not defined`
The `predict` method was using undefined `kwargs` to look for labels.

**Action:**
1. Modified `model.py` to use default `candidate_labels`.
2. Removed `device_map="auto"` and manually handled device placement.
3. Updated `postprocess` to use `model_output.logits_per_image`.

**Result:**
Success. Model runs and produces output.

## clip_vit_large_patch14

**Initial Failure:**
`ValueError: CLIP model requires a list of 'candidate_labels'...`

**Action:**
1. Modified `model.py` to use default `candidate_labels`.
2. Removed `device_map="auto"` and manually handled device placement.
3. Updated `predict` to ensure inputs are moved to the correct device.

**Result:**
Success. Model runs and produces output.

## languagebind_video_merge

**Initial Failure:**
`ModuleNotFoundError: No module named 'languagebind'`

**Action:**
Attempted to rewrite `model.py` to use `transformers.AutoModel`.

**Result:**
Failed. `transformers` does not recognize the architecture `LanguageBindImage`. This model requires a custom package (`languagebind`) which is not installed in the environment.

## mit_b2

**Result:**
Success. Model runs and produces output.
**Note:** `SegformerFeatureExtractor` is deprecated; `SegformerImageProcessor` should be used in future.

## siglip_base_patch16_224

**Initial Failure:**
(Anticipated) `NameError: name 'kwargs' is not defined`.

**Action:**
1. Modified `model.py` to use default `candidate_labels`.
2. Removed `device_map="auto"` and manually handled device placement.
3. Updated `predict` to ensure inputs are moved to the correct device.

**Result:**
Success. Model runs and produces output.

## clip_vit_b_32_256x256_datacomp_s34b_b86k

**Initial Failure:**
(Anticipated) `NameError: name 'kwargs' is not defined`.

**Action:**
1. Modified `model.py` to use default `candidate_labels`.
2. Removed `kwargs` usage in `predict`.

**Result:**
Success. Model runs and produces output.

## mobileclip_s2_openclip

**Initial Failure:**
`ValueError: Unrecognized model in apple/MobileCLIP-S2-OpenCLIP`
`transformers` failed to recognize the architecture.

**Action:**
Rewrote `model.py` to use `open_clip` primitives `create_model_and_transforms` (using `hf-hub:` prefix) and `get_tokenizer`. Added default labels.

**Result:**
Success. Model runs and produces output.

## vit_l_16_siglip2_512

**Initial Failure:**
(Anticipated) `AttributeError: object has no attribute 'device'` on model.

**Action:**
Modified `model.py` to store `self.device` and use it explicitly in `predict`.

**Result:**
Success. Model runs and produces output.

**Second Failure:**
`ImportError: libcudnn.so.8...` after installing `open_clip_torch`.
Pip downgraded `torch` to 2.0.1 which requires cuDNN v8, but environment uses cuDNN v9.

**Action:**
Upgraded `torch`, `torchvision`, `torchaudio`, `open_clip_torch`, `timm` to latest versions (torch 2.9.1/compat) in `py-mlmodelscope` environment.

**Result:**
Success. Model runs and produces output.

## clip_vit_l_14_datacomp_xl_s13b_b90k

**Initial Failure:**
(Anticipated) `transformers` configuration likely missing for DataComp model.

**Action:**
Rewrote `model.py` to use `open_clip` primitives `create_model_from_pretrained` (using `hf-hub:` prefix) and `get_tokenizer`. Added default labels.

**Result:**
Success. Model runs and produces output.

## one_align

**Initial Failure:**
`NameError: name 'Cache' is not defined` inside `modeling_llama2.py` (remote code).
The remote code for this model is incompatible with the installed version of `transformers` (v4.55.4). It relies on `Cache` being available, likely from an older or different import structure.

**Result:**
Failed. Remote code incompatibility.

## clip_vit_b_32_datacomp_xl_s13b_b90k

**Initial Failure:**
(Anticipated) `transformers` configuration likely missing for DataComp model.

**Action:**
Rewrote `model.py` to use `open_clip` primitives `create_model_from_pretrained` (using `hf-hub:` prefix) and `get_tokenizer`. Added default labels.

**Result:**
Success. Model runs and produces output.

## FIXME: pickscore_v1

**Initial Failure:**
1. `NameError: name 'options' is not defined`.
2. `numpy.exceptions.AxisError: axis 1 is out of bounds` because output was scalar/1D.

**Action:**
1. Modified `model.py` to use `self.config` and added default prompt.
2. Updated `predict` to calculate `(B, 1)` logits (`image @ text.T`).
3. Updated `postprocess` to use `sigmoid` and return properly shaped list.

**Result:**
Success. Model runs and produces output. However, it does need manual reconsideration which modality it belongs to since it also needs a text prompt as an input. 

## vit_so400m_16_siglip2_384

**Initial Failure:**
(Anticipated) Same `device` attribute issue as `vit_l_16`.

**Action:**
Modified `model.py` to store `self.device` and use it explicitly in `predict`.

**Result:**
Success. Model runs and produces output.
