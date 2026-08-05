import unittest

from mlmodelscope.outputprocessor import OutputProcessor


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


if __name__ == "__main__":
    unittest.main()
