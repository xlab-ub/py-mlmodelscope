from ....pytorch_abc import PyTorchAbstractClass
import os

import torch
import torchaudio
import torchaudio.transforms as T
from pyannote.audio import Pipeline


class PyTorch_Pyannote_Diarization(PyTorchAbstractClass):
  def __init__(self, config=None):
    super().__init__(config)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or self.config.get('huggingface_token')
    if not hf_token:
      raise ValueError(
          "Pyannote speaker diarization requires HUGGINGFACE_TOKEN or "
          "a huggingface_token model configuration value."
      )

    self.model = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    if self.model is None:
      raise RuntimeError(
            "Failed to load PyAnnote Pipeline. Verify your HUGGINGFACE_TOKEN "
            "is valid and that you have accepted the user conditions for "
            "'pyannote/speaker-diarization-3.1' on Hugging Face."
        )

    device = torch.device(self._device)
    self.model.to(device)
    self._is_dispatched = True

    self.mel_transform = T.MelSpectrogram(sample_rate=16000, n_mels=80)
    self.power_to_db = T.AmplitudeToDB(stype="power", top_db=80)

  def eval(self):
    # pyannote.audio.Pipeline is not an nn.Module.
    pass

  def to(self, device, multi_gpu=False):
    self.model.to(torch.device(device))
    self.device = device
    self._is_dispatched = True
    return self

  def preprocess(self, input_audios):
    preprocessed_inputs = []

    for file_path in input_audios:
        waveform, sample_rate = torchaudio.load(file_path)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        preprocessed_inputs.append({
            "waveform": waveform,
            "sample_rate": 16000
        })

    return preprocessed_inputs

  def predict(self, model_input):
    predictions = []
    with torch.no_grad():
      for audio_payload in model_input:
        captured = {}

        def capture_pipeline_artifact(step_name, artifact, **kwargs):
          if step_name == "discrete_diarization":
            captured["speaker_activity"] = artifact

        diarization = self.model(audio_payload, hook=capture_pipeline_artifact)
        predictions.append({
            "diarization": diarization,
            "waveform": audio_payload["waveform"],
            "sample_rate": audio_payload["sample_rate"],
            "speaker_activity": captured.get("speaker_activity")
        })
    return predictions

  def postprocess(self, model_output):
    final_diarisations = []

    for output in model_output:
        annotation = output["diarization"]
        waveform = output["waveform"]
        sample_rate = output["sample_rate"]
        speaker_activity = output.get("speaker_activity")
        speakers = annotation.labels()

        file_timeline = []
        for segment, track, speaker in annotation.itertracks(yield_label=True):
            confidence = self._segment_confidence(
                speaker_activity, segment, speaker, speakers
            )

            start_frame = int(segment.start * sample_rate)
            end_frame = int(segment.end * sample_rate)
            segment_waveform = waveform[:, start_frame:end_frame]

            result = {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "speaker": speaker,
                "confidence": confidence,
                "confidence_type": "mean_speaker_activity" if confidence is not None else None,
                "spectrogram": self._spectrogram(segment_waveform)
            }
            file_timeline.append(result)
        final_diarisations.append(file_timeline)

    return final_diarisations

  def _spectrogram(self, waveform, max_frames=512):
    if waveform.numel() == 0:
      return []

    spectrogram = self.power_to_db(self.mel_transform(waveform)).squeeze(0)
    if spectrogram.shape[-1] > max_frames:
      spectrogram = torch.nn.functional.adaptive_avg_pool1d(
          spectrogram, max_frames
      )
    return spectrogram.cpu().tolist()

  @staticmethod
  def _segment_confidence(speaker_activity, segment, speaker, speakers):
    if speaker_activity is None or speaker not in speakers:
      return None

    try:
      activity = speaker_activity.crop(segment, mode="center")
      activity = torch.as_tensor(activity, dtype=torch.float32)
      if activity.numel() == 0 or activity.ndim < 2:
        return None
      speaker_index = speakers.index(speaker)
      values = activity[..., speaker_index]
      values = values[torch.isfinite(values)]
      if values.numel() == 0:
        return None
      return round(float(values.mean().clamp(0.0, 1.0)), 3)
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
      return None
