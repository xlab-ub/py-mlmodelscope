from ....pytorch_abc import PyTorchAbstractClass 

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

class PyTorch_Transformers_Stable_Diffusion_v1_4(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    model_id = "CompVis/stable-diffusion-v1-4"
    self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", use_safetensors=True)
    self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", use_safetensors=True)
    self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", use_safetensors=True)
    self.scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")

    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) 
    
    self.height = self.config.get('height', self.unet.config.sample_size * self.vae_scale_factor) 
    self.width = self.config.get('width', self.unet.config.sample_size * self.vae_scale_factor) 
    self.num_inference_steps = self.config.get('num_inference_steps', 25)  # Number of denoising steps
    self.guidance_scale = self.config.get('guidance_scale', 7.5)  # Scale for classifier-free guidance
    self.seed = self.config.get('seed', 0)

  def preprocess(self, input_prompts): 
    batch_size = len(input_prompts)
    text_inputs = self.tokenizer(input_prompts, padding="max_length", max_length=self.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").input_ids.to(self.device)
    unconditional_input = self.tokenizer([""] * batch_size, padding="max_length",
                                         max_length=self.tokenizer.model_max_length, return_tensors="pt"
                                         ).input_ids.to(self.device)
    
    with torch.no_grad():
      # Encode all prompts in one go to leverage GPU parallelization
      cond_embeddings = self.text_encoder(text_inputs).last_hidden_state
      uncond_embeddings = self.text_encoder(unconditional_input).last_hidden_state

    # Concatenate conditional and unconditional embeddings
    model_input = torch.cat([uncond_embeddings, cond_embeddings], dim=0)

    self.latents = torch.randn(
      (batch_size, self.unet.config.in_channels, self.height // 8, self.width // 8),
      generator=self.generator, device=self.device
    ) * self.scheduler.init_noise_sigma

    self.scheduler.set_timesteps(self.num_inference_steps)

    return model_input 
  
  def predict(self, model_input): 
    for t in self.scheduler.timesteps: # tqdm is optional and can be removed for maximum performance 
      # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
      latent_model_input = torch.cat([self.latents] * 2)

      latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

      # predict the noise residual
      with torch.no_grad(): 
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=model_input).sample

      # perform guidance
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

      # compute the previous noisy sample x_t -> x_t-1
      self.latents = self.scheduler.step(noise_pred, t, self.latents).prev_sample
    
    return self.latents 

  def postprocess(self, model_output):
    # Decode latents into images
    image_latents = model_output * (1 / 0.18215)
    with torch.no_grad(): 
      images = self.vae.decode(image_latents).sample
    images = ((images / 2) + 0.5).clamp(0, 1)

    # Ensure we have a four-dimensional tensor after squeezing [B, C, H, W]
    if images.ndim == 3:
        images.unsqueeze_(0)

    # Convert to uint8 and prepare for output
    images = (images * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return [image.tolist() for image in images]

  def to(self, device):
    self.device = device 

    self.vae.to(device)
    self.text_encoder.to(device)
    self.unet.to(device)

    self.generator = torch.Generator(device=device).manual_seed(self.seed)

  def eval(self):
    self.vae.eval()
    self.text_encoder.eval()
    self.unet.eval()
