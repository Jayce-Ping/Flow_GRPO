# flowgrpo.diffusers_patch.flux_pipeline_kontext.py
from argparse import Namespace
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import torch
import numpy as np
import math
from typing import Optional, Union

from diffusers import FluxKontextPipeline
from diffusers.utils import logging
from diffusers.pipelines.flux.pipeline_flux import logger
from diffusers.image_processor import PipelineImageInput

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

from flow_grpo.utils import to_broadcast_tensor

def gaussian_log_prob(x, mean, var):
    return -((x - mean) ** 2) / (2 * var) - torch.log(torch.sqrt(var)) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

def denoising_sde_step_with_logprob(
    scheduler: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    sigma: Union[float, torch.FloatTensor],
    sigma_prev: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: Union[int, float, list[float], torch.FloatTensor] = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
    sigma_max: Optional[float] = 0.98,
    cps : bool = False,
    return_log_prob : bool = True,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Predict the sample from the previous timestep by **reversing** the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity). Specially, when noise_level is zero, the process becomes deterministic.

    Args:
        scheduler (`FlowMatchEulerDiscreteScheduler`):
            A scheduler object that handles the scheduling of the diffusion process.
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        sigma (`float` | `torch.FloatTensor`):
            The current noise level (sigma) in the diffusion chain. This can be different for each sample in the batch.
        sigma_prev (`float` | `torch.FloatTensor`):
            The previous noise level (sigma) in the diffusion chain. This can be different for each sample in the batch.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        noise_level (`int` | `float` | `list[float]` | `torch.FloatTensor`, *optional*, defaults to 0.7):
            The noise level parameter, can be different for each sample in the batch. This parameter controls the standard deviation of the noise added to the denoised sample.
        prev_sample (`torch.FloatTensor`):
            The next insance of the sample. If given, calculate the log_prob using given `prev_sample` as predicted value.
        generator (`torch.Generator`, *optional*):
            A random number generator for SDE solving. If not given, a random generator will be used.
        sigma_max (`float`, *optional*, defaults to 0.98):
            The maximum noise level (sigma) used to avoid numerical issues when sigma is 1.
        cps (`bool`, *optional*, defaults to False):
            Whether to use coefficient preserving sampling (CPS) in the denoising step.
        return_log_prob (`bool`, *optional*, defaults to True):
            Whether to return the log probability of the transition.
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    # Convert noise_level to a tensor with shape (batch_size, 1, 1)
    noise_level = to_broadcast_tensor(noise_level, sample)
    sigma = to_broadcast_tensor(sigma, sample)
    sigma_prev = to_broadcast_tensor(sigma_prev, sample)

    dt = sigma_prev - sigma # dt is negative, (batch_size, 1, 1)

    if not cps:
        sigma_max = to_broadcast_tensor(sigma_max, sample) # To avoid dividing by zero
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level # (batch_size, 1, 1)
        
        # FlowGRPO sde
        # Equation (9):
        #              sigma <-> t
        #        noise_level <-> a below Equation (9) - gives sigma_t = sqrt(t/(1-t))*a in the paper - corresponsds to std_dev_t = sqrt(sigma/(1-sigma))*noise_level here
        #                 dt <-> -\delta_t
        #       model_output <-> v_\theta(x_t, t)
        #             sample <-> x_t
        #        prev_sample <-> x_{t+\delta_t}
        #          std_dev_t <-> sigma_t

        prev_sample_mean = sample * (1 + std_dev_t**2 / (2 * sigma) * dt) + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        
        if prev_sample is None:
            # Non-determistic step, add noise to it
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            # Last term of Equation (9)
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

        if return_log_prob:
            log_prob = (
                -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
                - torch.log(std_dev_t * torch.sqrt(-1 * dt))
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    else:
        # FlowCPS
        std_dev_t = sigma_prev * torch.sin(noise_level * torch.pi / 2) # sigma_t in paper
        pred_original_sample = sample - sigma * model_output # predicted x_0 in paper
        noise_estimate = sample + model_output * (1 - sigma) # predicted x_1 in paper
        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)
    
        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        if return_log_prob:
            log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    if not return_log_prob:
        log_prob = torch.zeros(sample.shape[0], device=sample.device)

    # Returns x_{t+\delta_t}, log_prob, x_{t+\delta_t} mean, sigma_t
    return prev_sample, log_prob, prev_sample_mean, std_dev_t


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def set_scheduler_timesteps(
    scheduler,
    num_inference_steps: int,
    seq_len: int,
    sigmas: Optional[List[float]] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    # 5. Prepare scheduler, shift timesteps/sigmas according to image size (image_seq_len)
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
        sigmas = None

    mu = calculate_shift(
        seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    return timesteps


def compute_log_prob(
        transformer,
        pipeline : FluxKontextPipeline,
        sample : dict[str, torch.Tensor],
        timestep_index : int,
        config : Namespace
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    # 1. Prepare parameters
    latents = sample["all_latents"][:, timestep_index] # Latents at current timestep, shape (B, seq_len, C)
    next_latents = sample["all_latents"][:, timestep_index + 1] # Latents at next timestep, shape (B, seq_len, C)
    image_latents = sample["image_latents"] # Image latents, shape (B, image_seq_len, C)
    num_inference_steps = config.sample.num_steps
    scheduler = pipeline.scheduler
    timestep = sample["timesteps"][:, timestep_index] # (B,)
    timestep_next = sample["timesteps"][:, timestep_index + 1] if timestep_index + 1 < sample["timesteps"].shape[1] else torch.zeros_like(timestep) # (B,)
    timestep_max =  sample["timesteps"][:, 1]

    batch_size = latents.shape[0]
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    height = config.resolution if 'height' not in sample else sample['height'][0] # All height/width in the batch should be the same
    width = config.resolution if 'width' not in sample else sample['width'][0] # All height/width in the batch should be the same
    prompt = sample['prompt']
    device = latents.device
    dtype = latents.dtype

    # 1. Set the scheduler, shift timesteps/sigmas according to full image size (image_seq_len)
    _ = set_scheduler_timesteps(
        scheduler=pipeline.scheduler,
        num_inference_steps=num_inference_steps,
        seq_len=latents.shape[1],
        device=device,
    )
    noise_level = pipeline.scheduler.get_noise_levels()[timestep_index].item()

    # 2. Prepare prompt_embeds
    logger.setLevel(logging.ERROR) # To silent CLIP overflow warning
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device=device,
        max_sequence_length=config.max_sequence_length,
    )
    logger.setLevel(logging.WARNING) # Restore logger level
    

    # 3. Prepare image_ids according to the latents
    latent_ids = pipeline._prepare_latent_image_ids(
        batch_size=batch_size,
        height=height // (pipeline.vae_scale_factor * 2),
        width=width // (pipeline.vae_scale_factor * 2),
        device=device,
        dtype=dtype,
    )
    image_ids = pipeline._prepare_latent_image_ids(
        batch_size=batch_size,
        height=height // (pipeline.vae_scale_factor * 2),
        width=width // (pipeline.vae_scale_factor * 2),
        device=device,
        dtype=dtype,
    )
    image_ids[..., 0] = 1

    latent_ids = torch.cat([latent_ids, image_ids], dim=0)  # dim 0 is sequence dimension

    # 4. Concatenate image_latents to latents for Kontext
    latent_model_input = torch.cat([latents, image_latents], dim=1)

    # 5. Prepare guidance and predict the noise residual
    guidance = torch.tensor([config.sample.guidance_scale], device=device)

     # Predict the noise residual
    model_pred = transformer(
        hidden_states=latent_model_input,
        timestep=timestep / 1000, # which is scheduler.sigmas[timestep_index] exactly
        guidance=guidance.expand(latents.shape[0]),
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype),
        img_ids=latent_ids,
        return_dict=False,
    )[0]
    
    # 6. Extract only the latents part from the prediction (not the image_latents part)
    model_pred = model_pred[:, :latents.shape[1]]
    
    # 7. Compute log prob
    # Compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = denoising_sde_step_with_logprob(
        scheduler=scheduler,
        model_output=model_pred.float(),
        sigma=timestep / 1000,
        sigma_prev=timestep_next / 1000,
        sample=latents.float(),
        noise_level=noise_level,
        prev_sample=next_latents.float(),
        cps=config.sample.cps,
        return_log_prob=True,
        sigma_max=timestep_max / 1000
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

@torch.no_grad()
def flux_kontext_pipeline(
    pipeline : FluxKontextPipeline,
    image: Optional[PipelineImageInput] = None,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt: Union[str, List[str]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 3.5,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    max_area: int = 1024**2,
    noise_level: Optional[float] = None,
    cps : bool = False,
) -> Tuple[
        torch.FloatTensor,
        List[torch.FloatTensor],
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor

    original_height, original_width = height, width
    aspect_ratio = width / height
    width = round((max_area * aspect_ratio) ** 0.5)
    height = round((max_area / aspect_ratio) ** 0.5)

    multiple_of = pipeline.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    if height != original_height or width != original_width:
        logger.warning(
            f"Generation `height` and `width` have been adjusted to {height} and {width} to fit the model requirements."
        )

    # 1. Check inputs. Raise error if not correct
    pipeline.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    pipeline._guidance_scale = guidance_scale
    pipeline._joint_attention_kwargs = joint_attention_kwargs
    pipeline._current_timestep = None
    pipeline._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
        prompt = [prompt]
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if isinstance(generator, torch.Generator):
        generator = [generator] * batch_size

    device = pipeline._execution_device

    lora_scale = (
        pipeline.joint_attention_kwargs.get("scale", None)
        if pipeline.joint_attention_kwargs is not None else None
    )
    
    # 3. Encode prompts
    logger.setLevel(logging.ERROR) # To silent CLIP overflow warning
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    logger.setLevel(logging.WARNING) # Restore logger level

    # 4. Preprocess image and prepare image latents
    if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == pipeline.latent_channels):
        image = pipeline.image_processor.resize(image, height, width)
        image = pipeline.image_processor.preprocess(image, height, width)
    
    # 5. Prepare latent variables
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    latents, image_latents, latent_ids, image_ids = pipeline.prepare_latents(
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    if image_ids is not None:
        latent_ids = torch.cat([latent_ids, image_ids], dim=0)  # dim 0 is sequence dimension

    # 6. Prepare scheduler, shift timesteps/sigmas according to image size (image_seq_len)
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    if hasattr(pipeline.scheduler.config, "use_flow_sigmas") and pipeline.scheduler.config.use_flow_sigmas:
        sigmas = None

    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    # FlowMatchEulerDiscreteScheduler has order 1, which gives num_warmup_steps=0
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)
    pipeline._num_timesteps = len(timesteps)

    # handle guidance
    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)

    # 7. Denoising loop
    all_latents = [latents]
    all_noise_timestep_indices = []
    pipeline.scheduler.set_begin_index(0)
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            pipeline._current_timestep = t
            # Get noise_level. If not given in the arguments, use the sliding window scheduler's method to retrieve it.
            current_noise_level = noise_level if noise_level is not None else pipeline.scheduler.get_noise_level_for_timestep(t)

            # Concatenate image_latents to latents for Kontext
            latent_model_input = torch.cat([latents, image_latents], dim=1)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            timestep_next = timesteps[i + 1].expand(latents.shape[0]).to(latents.dtype) if i + 1 < len(timesteps) else torch.zeros_like(timestep)

            noise_pred = pipeline.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance.expand(latents.shape[0]),
                pooled_projections=pooled_prompt_embeds.to(latents.dtype),
                encoder_hidden_states=prompt_embeds.to(latents.dtype),
                txt_ids=text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs=pipeline.joint_attention_kwargs,
                return_dict=False,
            )[0]

            # Extract only the latents part from the prediction (not the image_latents part)
            noise_pred = noise_pred[:, :latents.shape[1]]
            noise_pred = noise_pred.to(prompt_embeds.dtype)
            latents_dtype = latents.dtype

            latents, _, _, _ = denoising_sde_step_with_logprob(
                scheduler=pipeline.scheduler,
                model_output=noise_pred.float(),
                sigma=timestep / 1000,
                sigma_prev=timestep_next / 1000,
                sample=latents.float(),
                noise_level=current_noise_level,
                prev_sample=None,
                sigma_max=timesteps[1].item() / 1000,
                cps=cps,
                return_log_prob=False,
            )
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            all_latents.append(latents)
            if current_noise_level > 0:
                all_noise_timestep_indices.append(i)
    
            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

    latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    latents = latents.to(dtype=pipeline.vae.dtype)
    images = pipeline.vae.decode(latents, return_dict=False)[0]
    images = pipeline.image_processor.postprocess(images, output_type=output_type)

    # Offload all models
    pipeline.maybe_free_model_hooks()

    timesteps = timesteps.unsqueeze(0).expand(batch_size, -1) # (batch_size, num_inference_steps)
    return images, all_latents, prompt_embeds, pooled_prompt_embeds, all_noise_timestep_indices, timesteps, image_latents