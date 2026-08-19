import unittest
from types import SimpleNamespace

from mlmodelscope.outputprocessor import OutputProcessor
from mlmodelscope.pytorch_agent.models.default.audio_diarization.pyannote_diarization.model import (
    PyTorch_Pyannote_Diarization,
)


class AudioDiarizationOutputTest(unittest.TestCase):
    def test_serializes_segments_as_text_feature(self):
        segments = [
            {"start": 0.0, "end": 1.25, "speaker": "SPEAKER_00"},
            {"start": 1.25, "end": 2.5, "speaker": "SPEAKER_01"},
        ]

        result = OutputProcessor.process_final_outputs_for_serialization(
            "audio_diarization", [segments]
        )

        feature = result[0]["responses"][0]["features"][0]
        self.assertEqual(feature, {"text": segments, "type": "TEXT"})

    def test_averages_assigned_speaker_activity_as_confidence(self):
        class SpeakerActivity:
            def crop(self, segment, mode):
                self.crop_args = (segment, mode)
                return [[0.7, 0.2], [0.9, 0.1]]

        activity = SpeakerActivity()
        segment = SimpleNamespace(start=0.0, end=1.0)

        confidence = PyTorch_Pyannote_Diarization._segment_confidence(
            activity, segment, "SPEAKER_00", ["SPEAKER_00", "SPEAKER_01"]
        )

        self.assertEqual(confidence, 0.8)
        self.assertEqual(activity.crop_args, (segment, "center"))

    def test_omits_confidence_when_pipeline_activity_is_unavailable(self):
        confidence = PyTorch_Pyannote_Diarization._segment_confidence(
            None, SimpleNamespace(start=0.0, end=1.0), "SPEAKER_00", ["SPEAKER_00"]
        )

        self.assertIsNone(confidence)


if __name__ == "__main__":
    unittest.main()
