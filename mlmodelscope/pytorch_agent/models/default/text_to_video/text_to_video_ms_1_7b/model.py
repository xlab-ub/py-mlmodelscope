from ....pytorch_abc import PyTorchAbstractClass 
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import torch



class PyTorch_Transformers_Text_To_Video_Ms_1_7b(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {} 
    model_id = "damo-vilab/text-to-video-ms-1.7b"
    self.width = 256
    self.height = 256
    self.model = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    self.num_inference_steps = 25
    self.model.scheduler = DPMSolverMultistepScheduler.from_config(self.model.scheduler.config)
    self.video_fps = 10
  
  def preprocess(self, input_texts):
    # optimize for GPU memory
    self.model.enable_model_cpu_offload()
    self.model.enable_vae_slicing()
    return input_texts
  
  def predict(self, model_input): 
    #  frames[0] will return the first video in the batch
    return self.model(model_input, num_inference_steps=self.num_inference_steps ,width=self.width, height=self.height).frames[0]

  def postprocess(self, model_output):
      
      video_paths = []
      video_path = export_to_video(model_output, fps=self.video_fps)
      video_paths.append(video_path)

      return video_paths

  def eval(self):
    pass 



