from ....pytorch_abc import PyTorchAbstractClass
import os

import torch
import torchaudio
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
        predictions.append(self.model(audio_payload))
    return predictions

  def postprocess(self, model_output):
    final_diarisations = []

    for annotation in model_output:
        file_timeline = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            file_timeline.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker
            })
        final_diarisations.append(file_timeline)

    return final_diarisations
