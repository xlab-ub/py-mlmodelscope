from ....pytorch_abc import PyTorchAbstractClass 

import torch 
from transformers import AutoProcessor, VideoMAEForVideoClassification 
import av 
import numpy as np

class PyTorch_Transformers_XCLIP_Base_Patch32(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    self.processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32") 
    self.model = VideoMAEForVideoClassification.from_pretrained("microsoft/xclip-base-patch32") 
  
  def read_video_pyav(self, container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
      container (`av.container.input.InputContainer`): PyAV container.
      indices (`List[int]`): List of frame indices to decode.
    Returns:
      result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
      if i > end_index:
        break
      if i >= start_index and i in indices:
        frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

  def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
      clip_len (`int`): Total number of frames to sample.
      frame_sample_rate (`int`): Sample every n-th frame.
      seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
      indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

  def preprocess(self, input_videos): 
    container = av.open(input_videos[0])
    indices = self.sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames) 
    video = self.read_video_pyav(container, indices) 
    pixel_values = self.processor(videos=list(video), return_tensors="pt").pixel_values
    return pixel_values 
    
  def predict(self, model_input): 
    return self.model(model_input) 
  
  def postprocess(self, model_output):
    probabilities = torch.nn.functional.softmax(model_output.logits, dim=1)
    return probabilities.tolist()