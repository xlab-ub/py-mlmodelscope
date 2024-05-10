from ....pytorch_abc import PyTorchAbstractClass 

import torch
from transformers import CLIPTextModel, CLIPTokenizer 
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerAncestralDiscreteScheduler 
from diffusers.image_processor import VaeImageProcessor 
from PIL import Image 

import inspect 
from typing import Optional 

class PyTorch_Transformers_Instruct_Pix2Pix(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    model_id = "timbrooks/instruct-pix2pix" 
    self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", use_safetensors=True)
    self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", use_safetensors=True)
    self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", use_safetensors=True) 
    self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    
    self.height = self.config.get('height', 512)  # default height of Stable Diffusion
    self.width = self.config.get('width', 512)  # default width of Stable Diffusion 
    self._num_inference_steps = self.config.get('num_inference_steps', 100)  # Number of denoising steps
    self._guidance_scale = self.config.get('guidance_scale', 7.5)  # Scale for classifier-free guidance
    self._image_guidance_scale = self.config.get('image_guidance_scale', 1.5) 
    self.num_images_per_prompt = self.config.get('num_images_per_prompt', 1) # The number of images to generate per prompt.
    self.eta = self.config.get('eta', 0.0)  # Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. 
                                            # Only applies to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
    self.seed = self.config.get('seed', 0)

  @property
  def guidance_scale(self):
      return self._guidance_scale

  @property
  def image_guidance_scale(self):
      return self._image_guidance_scale

  @property
  def num_timesteps(self):
      # return self._num_timesteps
      return self._num_inference_steps 

  @property
  def do_classifier_free_guidance(self):
    return self.guidance_scale > 1.0 and self.image_guidance_scale >= 1.0

  def retrieve_latents(
      self, encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
      ):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
      return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
      return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
      return encoder_output.latents
    else:
      raise AttributeError("Could not access latents of provided encoder_output")

  def prepare_extra_step_kwargs(self, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
      extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
    if accepts_generator:
      extra_step_kwargs["generator"] = generator
    return extra_step_kwargs

  def preprocess(self, input_images_and_prompts): 
    batch_size = len(input_images_and_prompts) 

    images = [Image.open(input_image_and_prompt[0]).convert('RGB').resize((self.height, self.width)) for input_image_and_prompt in input_images_and_prompts]
    prompts = [input_image_and_prompt[1] for input_image_and_prompt in input_images_and_prompts] 

    text_inputs = self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt")
    text_input_ids = text_inputs.input_ids.to(self.device) 
    
    if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
      attention_mask = text_inputs.attention_mask.to(self.device)
    else:
      attention_mask = None

    with torch.no_grad():
      prompt_embeds = self.text_encoder(text_input_ids, attention_mask=attention_mask)[0] 

    if self.text_encoder is not None:
      prompt_embeds_dtype = self.text_encoder.dtype
    else:
      prompt_embeds_dtype = self.unet.dtype

    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=self.device) 

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, self.num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * self.num_images_per_prompt, seq_len, -1)

    # get unconditional embeddings for classifier free guidance
    if self.do_classifier_free_guidance:
      uncond_input = self.tokenizer([""] * batch_size, padding="max_length",
                                    max_length=prompt_embeds.shape[1], truncation=True, return_tensors="pt")
      if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
        attention_mask = uncond_input.attention_mask.to(self.device)
      else:
        attention_mask = None

      with torch.no_grad(): 
        negative_prompt_embeds = self.text_encoder(
            uncond_input.input_ids.to(self.device),
            attention_mask=attention_mask,
        )
      negative_prompt_embeds = negative_prompt_embeds[0]
    
    if self.do_classifier_free_guidance:
      # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
      seq_len = negative_prompt_embeds.shape[1]

      negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=self.device)

      negative_prompt_embeds = negative_prompt_embeds.repeat(1, self.num_images_per_prompt, 1)
      negative_prompt_embeds = negative_prompt_embeds.view(batch_size * self.num_images_per_prompt, seq_len, -1)

      # For classifier free guidance, we need to do two forward passes.
      # Here we concatenate the unconditional and text embeddings into a single batch
      # to avoid doing two forward passes
      # pix2pix has two negative embeddings, and unlike in other pipelines latents are ordered [prompt_embeds, negative_prompt_embeds, negative_prompt_embeds]
      prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds, negative_prompt_embeds])

    images = self.image_processor.preprocess(images)

    self.scheduler.set_timesteps(self.num_timesteps, device=self.device)

    images = images.to(device=self.device, dtype=prompt_embeds.dtype)

    batch_size = batch_size * self.num_images_per_prompt

    if images.shape[1] == 4:
      image_latents = images
    else:
      image_latents = self.retrieve_latents(self.vae.encode(images), sample_mode="argmax")

    if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
      # expand image_latents for batch_size
      # deprecation_message = (
      #     f"You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial"
      #     " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
      #     " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
      #     " your script to pass as many initial images as text prompts to suppress this warning."
      # )
      # deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
      additional_image_per_prompt = batch_size // image_latents.shape[0]
      image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
    elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
      raise ValueError(
          f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
      )
    else:
      image_latents = torch.cat([image_latents], dim=0)

    if self.do_classifier_free_guidance:
      uncond_image_latents = torch.zeros_like(image_latents)
      image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

    self.image_latents = image_latents 

    height, width = image_latents.shape[-2:]
    height = height * self.vae_scale_factor
    width = width * self.vae_scale_factor

    num_channels_latents = self.vae.config.latent_channels 

    if isinstance(self.generator, list) and len(self.generator) != batch_size:
      raise ValueError(
        f"You have passed a list of generators of length {len(self.generator)}, but requested an effective batch"
        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    self.latents = torch.randn(
      (batch_size, num_channels_latents, int(height) // self.vae_scale_factor, int(width) // self.vae_scale_factor),
      generator=self.generator, device=self.device, dtype=prompt_embeds.dtype
      ) * self.scheduler.init_noise_sigma

    num_channels_image = image_latents.shape[1]
    if num_channels_latents + num_channels_image != self.unet.config.in_channels:
        raise ValueError(
            f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
            f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
            f" `num_channels_image`: {num_channels_image} "
            f" = {num_channels_latents+num_channels_image}. Please verify the config of"
            " `pipeline.unet` or your `image` input."
        )
    
    self.extra_step_kwargs = self.prepare_extra_step_kwargs(self.generator, self.eta) 

    return prompt_embeds 
  
  def predict(self, model_input): 
    for t in self.scheduler.timesteps: # tqdm is optional and can be removed for maximum performance 
      # Expand the latents if we are doing classifier free guidance.
      # The latents are expanded 3 times because for pix2pix the guidance\
      # is applied for both the text and the input image.
      latent_model_input = torch.cat([self.latents] * 3) if self.do_classifier_free_guidance else self.latents

      # concat latents, image_latents in the channel dimension
      scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
      scaled_latent_model_input = torch.cat([scaled_latent_model_input, self.image_latents], dim=1)

      # predict the noise residual
      with torch.no_grad(): 
        noise_pred = self.unet(
          scaled_latent_model_input,
          t,
          encoder_hidden_states=model_input,
          added_cond_kwargs=None,
          return_dict=False,
          )[0]

      # perform guidance
      if self.do_classifier_free_guidance:
        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
        noise_pred = (
          noise_pred_uncond
          + self.guidance_scale * (noise_pred_text - noise_pred_image)
          + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
          )

      # compute the previous noisy sample x_t -> x_t-1
      self.latents = self.scheduler.step(noise_pred, t, self.latents, **self.extra_step_kwargs, return_dict=False)[0]

    return self.latents 

  def postprocess(self, model_output):
    # TODO: Test self.image_processor.postprocess() 
    # https://github.com/huggingface/diffusers/blob/82be58c51272dcc7ebd5cbf8f48d444e3df96a1a/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py#L457 
    # Decode latents into images
    image_latents = model_output * (1 / self.vae.config.scaling_factor) 
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
