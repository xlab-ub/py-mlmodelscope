from ....pytorch_abc import PyTorchAbstractClass 
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet3DConditionModel, PNDMScheduler, DPMSolverMultistepScheduler
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from accelerate import Accelerator
from accelerate.hooks import remove_hook_from_module
# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp  
import torch



class PyTorch_Transformers_Text_To_Video_Ms_1_7b(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    model_id = "damo-vilab/text-to-video-ms-1.7b"
    self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", use_safetensors=True)\

    # MemoryEfficientAttentionFlashAttentionOp.enable()
    self.accelerator = Accelerator(mixed_precision="fp16")
    self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", use_safetensors=True)
    self.unet = UNet3DConditionModel.from_pretrained(model_id, subfolder="unet", use_safetensors=True)
    self.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    self.do_classifier_free_guidance = True
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) 
    self.height = self.config.get('height', self.unet.config.sample_size * self.vae_scale_factor) 
    self.width = self.config.get('width', self.unet.config.sample_size * self.vae_scale_factor) 
    self.num_inference_steps =  1 #self.config.get('num_inference_steps', 10)  # Number of denoising steps
    self.guidance_scale = self.config.get('guidance_scale', 7.5)  # Scale for classifier-free guidance
    self.seed = self.config.get('seed', 0)
    self.video_processor = VideoProcessor(do_resize=False, vae_scale_factor=self.vae_scale_factor)

  
  def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        batch_size, channels, num_frames, height, width = latents.shapepip
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        image = self.vae.decode(latents).sample
        video = image[None, :].reshape((batch_size, num_frames, -1) + image.shape[2:]).permute(0, 2, 1, 3, 4)
        video = video.float()
        return video
  
  def preprocess(self, input_prompts):
    self.vae, self.text_encoder, self.unet, self.scheduler = self.accelerator.prepare(self.vae, self.text_encoder, self.unet, self.scheduler)
    batch_size = len(input_prompts)
    text_inputs = self.tokenizer(input_prompts, padding="max_length", max_length=self.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").input_ids.to(self.device)
    unconditional_input = self.tokenizer([""] * batch_size, padding="max_length",
                                         max_length=self.tokenizer.model_max_length, return_tensors="pt"
                                         ).input_ids.to(self.device)
    
    # with torch.no_grad():
      # Encode all prompts in one go to leverage GPU parallelization
    cond_embeddings = self.text_encoder(text_inputs).last_hidden_state
    uncond_embeddings = self.text_encoder(unconditional_input).last_hidden_state
    # Concatenate conditional and unconditional embeddings
    model_input = torch.cat([uncond_embeddings, cond_embeddings], dim=0)

    self.latents = torch.randn(
      (batch_size, self.unet.config.in_channels, self.height // 8, self.width // 8),
      generator=self.generator, device=self.device
    ) * self.scheduler.init_noise_sigma

    self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)

    return model_input 

  
  def predict(self, model_input): 


    for i, t in enumerate(self.scheduler.timesteps): 

      latent_model_input = torch.cat([self.latents] * 2) 
      latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

      # predict the noise residual
    # with self.accelerator.autocast():
      noise_pred = self.unet(
          latent_model_input,
          t,
          encoder_hidden_states=model_input,
          return_dict=False,
      )[0]

      # perform guidance
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

      # reshape latents
      bsz, channel, frames, width, height = self.latents.shape
      self.latents = self.latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
      noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

      # compute the previous noisy sample x_t -> x_t-1
      self.latents = self.scheduler.step(noise_pred, t, self.latents).prev_sample

      # reshape latents back
      self.latents = self.latents.reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)
    #   # call the callback, if provided

    return self.latents 


  def postprocess(self, model_output):    # Decode latents into video
      video_tensor = self.decode_latents(model_output)
      video = self.video_processor.postprocess_video(video=video_tensor, output_type="np")

      return video

  def to(self, device):
    self.device = device 

    self.vae.to(device)
    self.text_encoder.to(device)
    # self.unet.to(device)
    self.generator = torch.Generator(device=device).manual_seed(self.seed)


  def eval(self):
    self.vae.eval()
    self.text_encoder.eval()
    self.unet.eval()



