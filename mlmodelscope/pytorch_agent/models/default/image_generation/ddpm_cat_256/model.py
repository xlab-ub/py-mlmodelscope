from ....pytorch_abc import PyTorchAbstractClass 

import torch
from diffusers import DDPMScheduler, UNet2DModel 

class PyTorch_Transformers_DDPM_Cat_256(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    model_id = "google/ddpm-cat-256"
    self.scheduler = DDPMScheduler.from_pretrained(model_id)
    self.model = UNet2DModel.from_pretrained(model_id, use_safetensors=True) 
    
    self.sample_size = self.model.config.sample_size

    self.num_inference_steps = self.config.get('num_inference_steps', 25) # num_inference_steps: int = 1000 

    self.scheduler.set_timesteps(self.num_inference_steps) 

  def preprocess(self, no_input): 
    return torch.randn((len(no_input), 3, self.sample_size, self.sample_size), device=self.device) 
  
  def predict(self, model_input): 
    noises = model_input 
    for t in self.scheduler.timesteps: 
      with torch.no_grad(): 
        noisy_residual = self.model(noises, t).sample

      previous_noisy_sample = self.scheduler.step(noisy_residual, t, noises).prev_sample
      noises = previous_noisy_sample 
    
    return noises 

  def postprocess(self, model_output):
    images = (model_output / 2 + 0.5).clamp(0, 1) 

    # Ensure we have a four-dimensional tensor after squeezing [B, C, H, W]
    if images.ndim == 3:
        images.unsqueeze_(0)

    images = (images * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return [image.tolist() for image in images]

  def to(self, device):
    self.device = device 

    self.model.to(device) 
