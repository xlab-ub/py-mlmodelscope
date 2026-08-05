import base64
import io
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


SCHEMA_VERSION = "1.3"
SUPPORTED_MODELS = {
    "torchvision_resnet_18",
    "torchvision_resnet_34",
    "torchvision_resnet_50",
    "torchvision_resnet_101",
    "torchvision_resnet_152",
}
SUPPORTED_TEXT_MODELS = {"gpt_2", "gpt2"}


def unsupported_explanation(reason):
    return {
        "schemaVersion": SCHEMA_VERSION,
        "status": "unsupported",
        "method": "grad_cam",
        "targetLayer": "layer4.-1",
        "classes": [],
        "comparison": None,
        "limitations": [reason],
    }


def unsupported_text_explanation(reason, top_k=5):
    return {
        "schemaVersion": SCHEMA_VERSION,
        "status": "unsupported",
        "method": "token_probability",
        "topK": top_k,
        "tokens": [],
        "pipeline": None,
        "limitations": [reason],
    }


def failed_text_explanation(message, top_k=5):
    return {
        "schemaVersion": SCHEMA_VERSION,
        "status": "failed",
        "method": "token_probability",
        "topK": top_k,
        "tokens": [],
        "pipeline": None,
        "limitations": [message],
    }


class TokenProbabilityExplainer:
    def __init__(self, model_name, task, multi_gpu=False):
        self.model_name = model_name.lower().replace(".", "_")
        self.task = task
        self.multi_gpu = multi_gpu

    def unsupported_reason(self, model_input):
        if self.task != "text_to_text":
            return "Token probability explanations support text-to-text generation only."
        if self.model_name not in SUPPORTED_TEXT_MODELS:
            return "Token probability explanations v1 support the PyTorch GPT-2 wrapper only."
        if self.multi_gpu:
            return "Token probability explanations v1 support single-device inference only."
        if not isinstance(model_input, Mapping) or "input_ids" not in model_input:
            return "Token probability explanations require tokenizer input IDs."
        if model_input["input_ids"].shape[0] != 1:
            return "Token probability explanations v1 require batch size one."
        return None

    def explain(self, wrapper, model_input, top_k=5):
        try:
            return wrapper.predict_with_scores(model_input, top_k=top_k)
        except Exception:
            with torch.no_grad():
                model_output = wrapper.predict(model_input)
            return (
                model_output,
                failed_text_explanation(
                    "The text was generated, but token probability explanation failed.",
                    top_k,
                ),
            )


class GradCAMExplainer:
    def __init__(self, model_name, task, multi_gpu=False):
        self.model_name = model_name.lower().replace(".", "_")
        self.task = task
        self.multi_gpu = multi_gpu

    def unsupported_reason(self, model_input):
        if self.task != "image_classification":
            return "Grad-CAM v1 supports image classification only."
        if self.model_name not in SUPPORTED_MODELS:
            return "Grad-CAM v1 supports torchvision ResNet-18/34/50/101/152 models only."
        if self.multi_gpu:
            return "Grad-CAM v1 supports single-device inference only."
        if not hasattr(model_input, "shape") or model_input.shape[0] != 1:
            return "Grad-CAM v1 requires batch size one."
        return None

    def explain(self, wrapper, model_input, top_k=2):
        activations = None

        def capture_activations(_module, _inputs, output):
            nonlocal activations
            activations = output

        target_layer = wrapper.model.layer4[-1]
        handle = target_layer.register_forward_hook(capture_activations)
        try:
            wrapper.model.zero_grad(set_to_none=True)
            logits = wrapper.predict(model_input)
            try:
                explanation = self._build_explanation(
                    logits, activations, model_input, wrapper.features, top_k
                )
            except Exception:
                explanation = {
                    "schemaVersion": SCHEMA_VERSION,
                    "status": "failed",
                    "method": "grad_cam",
                    "targetLayer": "layer4.-1",
                    "topK": top_k,
                    "classes": [],
                    "comparison": None,
                    "limitations": [
                        "The prediction completed, but Grad-CAM generation failed."
                    ],
                }
            return logits, explanation
        finally:
            handle.remove()
            wrapper.model.zero_grad(set_to_none=True)
            activations = None

    def _build_explanation(self, logits, activations, model_input, labels, top_k):
        if activations is None:
            raise RuntimeError("The target layer did not produce activations.")
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("Grad-CAM requires one classification output.")

        top_k = min(top_k, logits.shape[1])
        probabilities = torch.softmax(logits, dim=1)
        top_probabilities, top_indices = probabilities.topk(top_k, dim=1)
        model_input_image = self._model_input_to_image(model_input)
        classes = []

        for rank, (class_index, probability) in enumerate(
            zip(top_indices[0], top_probabilities[0]), start=1
        ):
            gradient = torch.autograd.grad(
                logits[0, class_index],
                activations,
                retain_graph=rank < top_k,
                create_graph=False,
            )[0]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            heatmap = torch.relu((weights * activations).sum(dim=1, keepdim=True))
            heatmap = F.interpolate(
                heatmap,
                size=model_input_image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            heatmap = self._normalize(heatmap).detach().cpu().numpy()
            heatmap_png, overlay_png, focus_mask_png = self._render_images(
                model_input_image, heatmap
            )
            index = int(class_index.item())
            classes.append(
                {
                    "rank": rank,
                    "index": index,
                    "label": labels[index] if labels and index < len(labels) else str(index),
                    "logit": float(logits[0, index].detach().cpu().item()),
                    "probability": float(probability.detach().cpu().item()),
                    "heatmap": {"mimeType": "image/png", "data": heatmap_png},
                    "overlay": {"mimeType": "image/png", "data": overlay_png},
                    "focusMask": {"mimeType": "image/png", "data": focus_mask_png},
                }
            )

        comparison = None
        if len(classes) >= 2:
            comparison = {
                "logitMargin": classes[0]["logit"] - classes[1]["logit"],
                "probabilityMargin": (
                    classes[0]["probability"] - classes[1]["probability"]
                ),
            }

        model_input_artifact = {
            "mimeType": "image/png",
            "data": self._encode_png(model_input_image),
        }
        pipeline_classes = [
            {
                "rank": class_result["rank"],
                "index": class_result["index"],
                "label": class_result["label"],
                "logit": class_result["logit"],
                "probability": class_result["probability"],
            }
            for class_result in classes
        ]

        return {
            "schemaVersion": SCHEMA_VERSION,
            "status": "complete",
            "method": "grad_cam",
            "targetLayer": "layer4.-1",
            "topK": top_k,
            "classes": classes,
            "comparison": comparison,
            "inputView": {
                "width": model_input_image.width,
                "height": model_input_image.height,
                "transform": "resize_shorter_side_256_center_crop_224",
                "description": (
                    "The overlay uses the exact 224 by 224 center-cropped pixels "
                    "passed to the model before normalization."
                ),
            },
            "pipeline": {
                "preprocess": {
                    "operations": [
                        "Convert image to RGB",
                        "Resize the shorter side to 256 pixels",
                        "Take the centered 224 by 224 crop",
                        "Convert pixels to a tensor",
                        "Normalize RGB channels with ImageNet mean and standard deviation",
                    ],
                    "modelInput": model_input_artifact,
                    "tensor": {
                        "shape": list(model_input.shape),
                        "dataType": str(model_input.dtype).replace("torch.", ""),
                        "normalization": {
                            "mean": [0.485, 0.456, 0.406],
                            "standardDeviation": [0.229, 0.224, 0.225],
                        },
                    },
                },
                "inference": {
                    "model": self.model_name,
                    "targetLayer": "layer4.-1",
                    "rawOutput": {
                        "type": "logits",
                        "shape": list(logits.shape),
                        "classes": pipeline_classes,
                    },
                },
                "postprocess": {
                    "operation": "softmax",
                    "description": (
                        "Softmax converts all class logits into probabilities "
                        "that sum to one, then the classes are ranked."
                    ),
                    "classes": pipeline_classes,
                },
                "finalOutput": pipeline_classes[0] if pipeline_classes else None,
            },
            "limitations": [
                "Highlighted regions influenced the selected class score; they do not prove object recognition, causality, or human-like attention.",
                "Grad-CAM is low resolution and may omit evidence used elsewhere in the network.",
            ],
        }

    @staticmethod
    def _normalize(heatmap):
        minimum = heatmap.min()
        maximum = heatmap.max()
        if torch.isclose(maximum, minimum):
            return torch.zeros_like(heatmap)
        return (heatmap - minimum) / (maximum - minimum)

    @staticmethod
    def _model_input_to_image(model_input):
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=model_input.device,
            dtype=model_input.dtype,
        ).view(3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=model_input.device,
            dtype=model_input.dtype,
        ).view(3, 1, 1)
        pixels = model_input[0].detach() * std + mean
        pixels = pixels.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((pixels * 255).round().astype(np.uint8), mode="RGB")

    @staticmethod
    def _render_images(original, heatmap):
        strength = np.clip((heatmap - 0.1) / 0.9, 0, 1)
        dark = np.array([35, 39, 52], dtype=np.float32)
        cyan = np.array([0, 210, 255], dtype=np.float32)
        magenta = np.array([255, 0, 210], dtype=np.float32)
        white = np.array([255, 255, 255], dtype=np.float32)

        cyan_mix = np.clip(strength / 0.45, 0, 1)[..., None]
        colored = dark * (1 - cyan_mix) + cyan * cyan_mix
        magenta_mix = np.clip((strength - 0.45) / 0.35, 0, 1)[..., None]
        colored = colored * (1 - magenta_mix) + magenta * magenta_mix
        peak_mix = np.clip((strength - 0.8) / 0.2, 0, 1)[..., None]
        colored = colored * (1 - peak_mix) + white * peak_mix
        colored = colored.astype(np.uint8)
        heatmap_image = Image.fromarray(colored, mode="RGB")

        grayscale = np.asarray(original.convert("L"), dtype=np.float32)
        context = np.stack((grayscale, grayscale, grayscale), axis=-1) * 0.72
        alpha = (np.power(strength, 0.7) * 0.88)[..., None]
        overlay_array = context * (1 - alpha) + colored.astype(np.float32) * alpha

        strong = strength >= 0.55
        interior = (
            strong
            & np.roll(strong, 1, axis=0)
            & np.roll(strong, -1, axis=0)
            & np.roll(strong, 1, axis=1)
            & np.roll(strong, -1, axis=1)
        )
        boundary = strong & ~interior
        overlay_array[boundary] = white
        overlay = Image.fromarray(
            np.clip(overlay_array, 0, 255).astype(np.uint8), mode="RGB"
        )

        positive_strength = strength[strength > 0]
        threshold = (
            max(0.38, float(np.percentile(positive_strength, 65)))
            if positive_strength.size
            else 1.0
        )
        focus = strength >= threshold
        if not focus.any() and positive_strength.size:
            focus = strength == strength.max()

        dilated = focus.copy()
        for _ in range(3):
            dilated |= (
                np.roll(dilated, 1, axis=0)
                | np.roll(dilated, -1, axis=0)
                | np.roll(dilated, 1, axis=1)
                | np.roll(dilated, -1, axis=1)
            )
        contour = dilated & ~focus

        original_array = np.asarray(original, dtype=np.float32)
        muted = np.asarray(
            original.convert("L").filter(ImageFilter.GaussianBlur(radius=2)),
            dtype=np.float32,
        )
        muted = np.stack((muted, muted, muted), axis=-1) * 0.28
        focus_view = muted
        focus_view[focus] = original_array[focus]
        focus_view[contour] = cyan
        focus_mask = Image.fromarray(
            np.clip(focus_view, 0, 255).astype(np.uint8), mode="RGB"
        )
        return (
            GradCAMExplainer._encode_png(heatmap_image),
            GradCAMExplainer._encode_png(overlay),
            GradCAMExplainer._encode_png(focus_mask),
        )

    @staticmethod
    def _encode_png(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
