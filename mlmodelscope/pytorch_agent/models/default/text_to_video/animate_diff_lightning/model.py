from ....pytorch_abc import PyTorchAbstractClass 
from diffusers import AutoencoderKL, UNet2DConditionModel, UNetMotionModel, MotionAdapter, DDIMScheduler, EulerDiscreteScheduler
from transformers import  CLIPTextModel, CLIPTokenizer
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from diffusers.video_processor import VideoProcessor
import torch

class PyTorch_Transformers_Animate_Diff(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    self.model_id = "emilianJR/epiCRealism" 
    self.step = 4  # Options: [1,2,4,8]
    self.ckpt = f"animatediff_lightning_{self.step}step_diffusers.safetensors"

    self.num_images_per_prompt = 1
    self.dtype = torch.float16

    self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
    self.text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder="text_encoder", use_safetensors=True)
    self.vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae", use_safetensors=True)
    self.unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet", use_safetensors=True)
                                            
    self.scheduler = EulerDiscreteScheduler(beta_schedule="linear", timestep_spacing="trailing")
    self.adapter = MotionAdapter()

    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) 
    self.height = self.config.get('height', self.unet.config.sample_size * self.vae_scale_factor) 
    self.width = self.config.get('width', self.unet.config.sample_size * self.vae_scale_factor) 
    self.num_inference_steps =  25 #self.config.get('num_inference_steps', 10)  # Number of denoising steps
    self.guidance_scale = self.config.get('guidance_scale', 7.5)  # Scale for classifier-free guidance
    self.num_frames = 10
    self.seed = self.config.get('seed', 0)
    self.video_processor = VideoProcessor(do_resize=False, vae_scale_factor=self.vae_scale_factor)

  
  def preprocess(self, input_prompts):
    with torch.no_grad():
      batch_size = len(input_prompts)

      text_inputs = self.tokenizer(input_prompts, padding="max_length", max_length=self.tokenizer.model_max_length, 
                                   truncation=True, return_tensors="pt")
      text_input_ids = text_inputs.input_ids
      
      prompt_embeds = self.text_encoder(text_input_ids.to(self.device), attention_mask=None)
      prompt_embeds = prompt_embeds[0]
      prompt_embeds_dtype = self.text_encoder.dtype

      bs_embed, seq_len, _ = prompt_embeds.shape
      
      # duplicate text embeddings for each generation per prompt, using mps friendly method
      prompt_embeds = prompt_embeds.repeat(1, self.num_images_per_prompt, 1)
      prompt_embeds = prompt_embeds.view(bs_embed * self.num_images_per_prompt, seq_len, -1)


      uncond_tokens = [""] * batch_size

      max_length = prompt_embeds.shape[1]
      
      uncond_input = self.tokenizer(
                  uncond_tokens,
                  padding="max_length",
                  max_length=max_length,
                  truncation=True,
                  return_tensors="pt",
              )
      
      negative_prompt_embeds = self.text_encoder(
          uncond_input.input_ids.to(self.device),
          attention_mask=None,
      )
      negative_prompt_embeds = negative_prompt_embeds[0]
      seq_len = negative_prompt_embeds.shape[1]

      negative_prompt_embeds = negative_prompt_embeds.repeat(1, self.num_images_per_prompt, 1)
      negative_prompt_embeds = negative_prompt_embeds.view(batch_size * self.num_images_per_prompt, seq_len, -1)

      # Concatenate conditional and unconditional embeddings
      prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

      shape = (
              batch_size,
              self.unet.config.in_channels,
              self.num_frames,
              self.height // self.vae_scale_factor,
              self.width // self.vae_scale_factor,
      )
      self.latents = randn_tensor(shape, generator=self.generator, device=torch.device(self.device), dtype=prompt_embeds_dtype)

      self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)

    return prompt_embeds 


  def predict(self, model_input):
    with torch.no_grad():
      for t in self.scheduler.timesteps:
          
          latent_model_input = torch.cat([self.latents] * 2)
          latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

          # predict the noise residual
          with torch.no_grad(): 
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=model_input,
                return_dict=False,
            )[0]

          # perform guidance
          noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
          noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

          # compute the previous noisy sample x_t -> x_t-1
          self.latents = self.scheduler.step(noise_pred, t, self.latents).prev_sample

    return self.latents  # Return latents without reshaping here


  def postprocess(self, model_output):    # Decode latents into video
      video_paths = []
      latents = 1 / self.vae.config.scaling_factor * model_output

      batch_size, channels, num_frames, height, width = latents.shape
      latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

      image = self.vae.decode(latents).sample
      video_tensor = image[None, :].reshape((batch_size, num_frames, -1) + image.shape[2:]).permute(0, 2, 1, 3, 4).float()
      # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
      video_frames = self.video_processor.postprocess_video(video=video_tensor, output_type="pil")[0]
      video_path = export_to_gif(video_frames)
      video_paths.append(video_path)

      return video_paths
  

  def to(self, device):
    self.device = device
    self.adapter.to(device, self.dtype)
    self.adapter.load_state_dict(load_file(hf_hub_download("ByteDance/AnimateDiff-Lightning",self.ckpt), device=device)) 
    self.vae.to(device)
    self.text_encoder.to(device)
    self.unet = UNetMotionModel.from_unet2d(self.unet, self.adapter)
    self.unet.to(device)
    self.generator = torch.Generator(device=device).manual_seed(self.seed)



  def eval(self):
    self.vae.eval()
    self.text_encoder.eval()
    self.unet.eval()
    


