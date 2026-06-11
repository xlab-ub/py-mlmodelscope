import base64
import io
import unittest
from unittest import mock

import torch
from PIL import Image

from mlmodelscope.pytorch_agent.explainability import (
    GradCAMExplainer,
    unsupported_explanation,
)


class TinyResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(4, 3)

    def forward(self, inputs):
        activations = self.layer4(inputs)
        return self.fc(self.pool(activations).flatten(1))


class Wrapper:
    def __init__(self):
        self.model = TinyResNet()
        self.features = ["zero", "one", "two"]

    def predict(self, model_input):
        return self.model(model_input)


class GradCAMExplainerTest(unittest.TestCase):
    def setUp(self):
        self.wrapper = Wrapper()
        self.explainer = GradCAMExplainer(
            "torchvision_resnet_18", "image_classification"
        )
    def test_builds_top_two_explanations_and_removes_hook(self):
        model_input = torch.rand(1, 3, 224, 224)
        target_layer = self.wrapper.model.layer4[-1]
        hook_count = len(target_layer._forward_hooks)

        logits, explanation = self.explainer.explain(
            self.wrapper, model_input, top_k=2
        )

        self.assertEqual(explanation["status"], "complete")
        self.assertEqual(explanation["topK"], 2)
        self.assertEqual(len(explanation["classes"]), 2)
        self.assertEqual(explanation["inputView"]["width"], 224)
        self.assertEqual(explanation["inputView"]["height"], 224)
        pipeline = explanation["pipeline"]
        self.assertEqual(pipeline["preprocess"]["tensor"]["shape"], [1, 3, 224, 224])
        self.assertEqual(pipeline["inference"]["rawOutput"]["type"], "logits")
        self.assertEqual(pipeline["postprocess"]["operation"], "softmax")
        self.assertEqual(
            pipeline["finalOutput"]["index"], explanation["classes"][0]["index"]
        )
        model_input_bytes = base64.b64decode(
            pipeline["preprocess"]["modelInput"]["data"]
        )
        self.assertEqual(Image.open(io.BytesIO(model_input_bytes)).size, (224, 224))
        self.assertEqual(len(target_layer._forward_hooks), hook_count)

        probabilities = torch.softmax(logits, dim=1)
        expected_probabilities, expected_indices = probabilities.topk(2, dim=1)
        for rank, class_result in enumerate(explanation["classes"]):
            self.assertEqual(class_result["index"], expected_indices[0, rank].item())
            self.assertAlmostEqual(
                class_result["probability"],
                expected_probabilities[0, rank].item(),
                places=6,
            )
            for artifact in ("heatmap", "overlay", "focusMask"):
                decoded = base64.b64decode(class_result[artifact]["data"])
                rendered = Image.open(io.BytesIO(decoded))
                self.assertEqual(rendered.size, (224, 224))
                self.assertEqual(class_result[artifact]["mimeType"], "image/png")

        self.assertGreaterEqual(explanation["comparison"]["probabilityMargin"], 0)
        self.assertAlmostEqual(
            explanation["comparison"]["logitMargin"],
            explanation["classes"][0]["logit"]
            - explanation["classes"][1]["logit"],
            places=6,
        )

    def test_reports_unsupported_constraints(self):
        self.assertIsNotNone(
            GradCAMExplainer("other_model", "image_classification").unsupported_reason(
                torch.rand(1, 3, 16, 16)
            )
        )
        self.assertIsNotNone(
            self.explainer.unsupported_reason(torch.rand(2, 3, 16, 16))
        )
        self.assertIsNotNone(
            GradCAMExplainer(
                "torchvision_resnet_18", "image_classification", multi_gpu=True
            ).unsupported_reason(torch.rand(1, 3, 16, 16))
        )
        self.assertEqual(unsupported_explanation("reason")["status"], "unsupported")
        normalized = self.explainer._normalize(torch.tensor([[2.0, 4.0]]))
        self.assertEqual(normalized.min().item(), 0)
        self.assertEqual(normalized.max().item(), 1)

    def test_overlay_base_image_comes_from_exact_model_tensor(self):
        pixels = torch.tensor([0.2, 0.4, 0.8]).view(1, 3, 1, 1)
        pixels = pixels.expand(1, 3, 224, 224)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        normalized_input = (pixels - mean) / std

        image = self.explainer._model_input_to_image(normalized_input)

        self.assertEqual(image.size, (224, 224))
        self.assertEqual(image.getpixel((0, 0)), (51, 102, 204))

    def test_overlay_desaturates_context_and_uses_contrast_evidence_colors(self):
        original = Image.new("RGB", (4, 4), color=(240, 180, 20))
        heatmap = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.6, 0.6, 0.0],
            [0.0, 0.6, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]).numpy()

        _heatmap_png, overlay_png, focus_mask_png = self.explainer._render_images(
            original, heatmap
        )
        overlay = Image.open(io.BytesIO(base64.b64decode(overlay_png)))
        focus_mask = Image.open(io.BytesIO(base64.b64decode(focus_mask_png)))

        low_evidence = overlay.getpixel((0, 0))
        high_evidence = overlay.getpixel((2, 2))
        self.assertEqual(low_evidence[0], low_evidence[1])
        self.assertEqual(low_evidence[1], low_evidence[2])
        self.assertGreaterEqual(high_evidence[2], high_evidence[1])
        self.assertLess(sum(focus_mask.getpixel((0, 0))), sum(original.getpixel((0, 0))))
        self.assertEqual(focus_mask.size, original.size)

    def test_explanation_failure_preserves_logits_and_removes_hook(self):
        target_layer = self.wrapper.model.layer4[-1]
        hook_count = len(target_layer._forward_hooks)

        with mock.patch.object(
            self.explainer,
            "_model_input_to_image",
            side_effect=RuntimeError("rendering failed"),
        ):
            logits, explanation = self.explainer.explain(
                self.wrapper, torch.rand(1, 3, 224, 224), top_k=2
            )

        self.assertEqual(logits.shape, (1, 3))
        self.assertEqual(explanation["status"], "failed")
        self.assertEqual(len(target_layer._forward_hooks), hook_count)


if __name__ == "__main__":
    unittest.main()
