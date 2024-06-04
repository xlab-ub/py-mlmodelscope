from ....pytorch_abc import PyTorchAbstractClass 
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch



class PyTorch_Transformers_Text_To_Video_Ms_1_7b(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    self.step = 4  # Options: [1,2,4,8]
    self.repo = "ByteDance/AnimateDiff-Lightning"
    self.ckpt = f"animatediff_lightning_{self.step}step_diffusers.safetensors"
    self.base = "emilianJR/epiCRealism"  #base model
    self.video_fps = 10
    self.adapter = MotionAdapter()
    self.guidance_scale = 1.0 
    self.dtype = torch.float16
    self.model = AnimateDiffPipeline.from_pretrained(self.base, motion_adapter=self.adapter, torch_dtype=self.dtype)
    self.model.scheduler = EulerDiscreteScheduler.from_config(self.model.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
  
  def preprocess(self, input_texts):
    # optimize for GPU memory
    self.model.enable_model_cpu_offload()
    self.model.enable_vae_slicing()
    return input_texts
  
  def predict(self, model_input): 
    return self.model(prompt="A girl smiling", guidance_scale=self.guidance_scale, num_inference_steps=self.step)

  def postprocess(self, model_output):
      gif_paths = []
      gif_path = export_to_gif(model_output.frames[0])
      gif_paths.append(gif_path)
      return gif_paths


  def to(self, device):
    self.device = device 
    self.adapter.to(device, self.dtype)
    self.adapter.load_state_dict(load_file(hf_hub_download(self.repo ,self.ckpt), device=device))
    self.model.to(device)

  def eval(self):
    pass 



