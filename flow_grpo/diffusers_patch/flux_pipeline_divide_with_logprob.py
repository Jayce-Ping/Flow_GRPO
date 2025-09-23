# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py
from argparse import Namespace
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import torch
import numpy as np
import math
from typing import Optional, Union

from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from ..scheduler import FlowMatchSlidingWindowScheduler, FlowMatchSubfigScheduler
from .denoising_step_with_logprob import denoising_sde_step_with_logprob
from flow_grpo.utils import divide_prompt

def divide_latents(latents, H, W, h, w):
    """
    Divide latents into sub-latents based on the specified sub-image size (h, w).
    Args:
        latents (torch.Tensor): The input latents tensor of shape (B, seq_len, C).
        H (int): Height of the original image.
        W (int): Width of the original image.
        h (int): Height of each sub-image.
        w (int): Width of each sub-image.

    Returns:
        torch.Tensor: A tensor of sub-latents of shape (B, rows, cols, sub_seq_len, C).
    """
    batch_size, image_seq_len, channels = latents.shape
    assert H % h == 0 and W % w == 0, "H and W must be divisible by h and w respectively."
    
    # Compute downsampling factor
    total_pixels = H * W
    downsampling_factor = total_pixels // image_seq_len

    # Check if downsampling factor is a perfect square
    downsample_ratio = int(math.sqrt(downsampling_factor))
    if downsample_ratio * downsample_ratio != downsampling_factor:
        raise ValueError(f"The downsampling ratio cannot be determined. Image pixels {total_pixels} and sequence length {image_seq_len} do not match.")
    
    # Calculate latent dimensions
    latent_H = H // downsample_ratio
    latent_W = W // downsample_ratio
    latent_h = h // downsample_ratio
    latent_w = w // downsample_ratio
    
    # Match check
    assert latent_H * latent_W == image_seq_len, f"Calculated latent dimensions {latent_H}x{latent_W} do not match sequence length {image_seq_len}"
    
    rows = latent_H // latent_h
    cols = latent_W // latent_w
    
    # Reshape latents to (B, latent_H, latent_W, C)
    latents = latents.view(batch_size, latent_H, latent_W, channels)
    
    sub_latents = [
        [
            latents[:, i * latent_h:(i + 1) * latent_h, j * latent_w:(j + 1) * latent_w, :].reshape(batch_size, -1, channels)
            for j in range(cols)
        ]
        for i in range(rows)
    ]

    # Since all sub-latents have the same shape, we can return them as a tensor
    sub_latents = torch.stack([torch.stack(row, dim=1) for row in sub_latents], dim=1)  # Shape: (B, rows, cols, sub_seq_len, C)
    
    return sub_latents


def merge_latents(sub_latents, H, W, h, w):
    """
    Merge sub-latents back into the original latents tensor.
    Args:
        sub_latents (torch.Tensor): A tensor of sub-latents of shape (B, rows, cols, sub_seq_len, C).
        H (int): Height of the original image.
        W (int): Width of the original image.
        h (int): Height of each sub-image.
        w (int): Width of each sub-image.
    Returns:
        torch.Tensor: The merged latents tensor of shape (B, seq_len, C).
    """
    batch_size, rows, cols, sub_seq_len, channels = sub_latents.shape
    
    vae_scale_factor = int(math.sqrt(h * w // sub_seq_len))
    # Calculate latent dimensions using the explicit parameters
    latent_h = h // vae_scale_factor
    latent_w = w // vae_scale_factor
    latent_H = H // vae_scale_factor
    latent_W = W // vae_scale_factor
    
    # Verify dimensions match
    assert latent_h * latent_w == sub_seq_len, f"sub_seq_len {sub_seq_len} does not match calculated sub-latent size {latent_h}x{latent_w}"
    assert rows * cols == (latent_H // latent_h) * (latent_W // latent_w), f"Grid size {rows}x{cols} does not match expected grid size"
    
    # Reshape sub_latents to (B, rows, cols, latent_h, latent_w, C)
    sub_latents = sub_latents.view(batch_size, rows, cols, latent_h, latent_w, channels)
    
    # Merge by rearranging dimensions
    # (B, rows, cols, latent_h, latent_w, C) -> (B, rows, latent_h, cols, latent_w, C)
    merged = sub_latents.permute(0, 1, 3, 2, 4, 5).contiguous()
    
    # Reshape to (B, latent_H, latent_W, C)
    merged = merged.view(batch_size, latent_H, latent_W, channels)
    
    # Final reshape to (B, seq_len, C)
    merged = merged.view(batch_size, latent_H * latent_W, channels)
    
    return merged

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


def compute_log_prob(
        transformer : FluxTransformer2DModel,
        pipeline : FluxPipeline,
        sample : dict[str, torch.Tensor],
        j : int,
        config : Namespace
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    # 1. Prepare parameters
    latents = sample["latents"][:, j]
    num_inference_steps = config.sample.num_steps
    scheduler = pipeline.scheduler
    timestep_index = sample['timestep_indices'][j] # timestep index in the scheduler.timesteps

    batch_size = latents.shape[0]
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    height = sample.get("height", config.resolution)
    width = sample.get("width", config.resolution)
    device = latents.device
    dtype = latents.dtype

    # 2. Prepare image_ids
    latents, image_ids = pipeline.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator=None,
        latents=latents
    )
    # 3. Set the scheduler, shift timesteps/sigmas according to image size (image_seq_len)
    sigmas_unshifted = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    if hasattr(pipeline.scheduler.config, "use_flow_sigmas") and pipeline.scheduler.config.use_flow_sigmas:
        # FluxPipeline.scheduler is FlowMatchEulerDiscreteScheduler, which has no such attribute, so sigmas_unshifted=None it is
        sigmas_unshifted = None

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
        sigmas=sigmas_unshifted,
        mu=mu,
    )
    timestep = timesteps[timestep_index]
    # This is for more general purpose of noise_level constrol, like decreasing noise_level across timesteps
    # If not, it should be always equal to config.sample.noise_level
    noise_level = pipeline.scheduler.get_noise_level_for_timestep(timestep)

    # 4. Prepare guidance and predict the noise residual
    if transformer.module.config.guidance_embeds:
        guidance = torch.tensor([config.sample.guidance_scale], device=device)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

     # Predict the noise residual
    model_pred = transformer(
        hidden_states=latents,
        timestep=timestep.expand(latents.shape[0]) / 1000, # which is scheduler.sigmas[timestep_index] exactly
        guidance=guidance,
        pooled_projections=sample["pooled_prompt_embeds"],
        encoder_hidden_states=sample["prompt_embeds"],
        txt_ids=torch.zeros(sample["prompt_embeds"].shape[1], 3).to(device=device, dtype=dtype),
        img_ids=image_ids,
        return_dict=False,
    )[0]
    
    # 5. Compute log prob
    # Compute the log prob of next_latents given latents under the current model
    # Here, use determistic denoising for normal diffusion process.
    prev_sample, log_prob, prev_sample_mean, std_dev_t = denoising_sde_step_with_logprob(
        scheduler=pipeline.scheduler,
        model_output=model_pred.float(),
        timestep=timestep.unsqueeze(0).repeat(latents.shape[0]),
        sample=latents.float(),
        noise_level=noise_level,
        prev_sample=sample["next_latents"][:, j].float(),
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t


@torch.no_grad()
def pipeline_with_logprob(
    pipeline : FluxPipeline,
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
    noise_level: Optional[float] = None,
    layout: Optional[Tuple[int, int]] = None,
) -> Tuple[
        torch.FloatTensor,
        List[torch.FloatTensor],
        List[torch.FloatTensor],
        List[int]
    ]:
    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor

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
    
    # Encode the entire prompt
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

    if layout is not None:
        # Encode each sub-prompt if layout is given
        sub_height = height // layout[0]
        sub_width = width // layout[1]
        divided_prompts = [divide_prompt(p) for p in prompt] # List (`length=batch_size`) of List[str] (`length=rows*cols + 1`)
        sub_prompts = sum([p[1:] for p in divided_prompts], []) # List of str, length = batch_size*rows*cols
        # Encode sub-prompts
        sub_prompt_embeds, sub_pooled_prompt_embeds, sub_text_ids = pipeline.encode_prompt(
            prompt=sub_prompts,
            prompt_2=sub_prompts,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = pipeline.transformer.config.in_channels // 4
        latents, latent_image_ids = pipeline.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Prepare latents for subfig
        _, sub_latent_image_ids = pipeline.prepare_latents(
            batch_size=batch_size * layout[0] * layout[1],
            num_channels_latents=num_channels_latents,
            height=sub_height,
            width=sub_width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=None,
        )

    # 5. Prepare scheduler, shift timesteps/sigmas according to image size (image_seq_len)
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
    if pipeline.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    # 6. Denoising loop
    all_latents = []
    all_log_probs = []
    all_timestep_indices = []
    pipeline.scheduler.set_begin_index(0)
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            pipeline._current_timestep = t
            # Get noise_level. If not given in the arguments, use the sliding window scheduler's method to retrieve it.
            current_noise_level = noise_level if noise_level is not None else pipeline.scheduler.get_noise_level_for_timestep(t)
            if current_noise_level > 0:
                all_latents.append(latents)

            if layout is not None and i < pipeline.scheduler.merge_step:
                # use sub-prompts and sub-latents if layout is given and not yet merged
                current_prompt_embeds = sub_prompt_embeds
                current_pooled_prompt_embeds = sub_pooled_prompt_embeds
                latents = divide_latents(latents, height, width, sub_height, sub_width) # (B, rows, cols, sub_seq_len, C)
                latents = latents.view(-1, latents.shape[3], latents.shape[4]) # (B*rows*cols, sub_seq_len, C)
                img_ids = sub_latent_image_ids
            else:
                current_prompt_embeds = prompt_embeds
                current_pooled_prompt_embeds = pooled_prompt_embeds
                img_ids = latent_image_ids

            # print("Latents shape:", latents.shape)
            # print("Img_ids shape:", img_ids.shape)
            # print("Prompt_embeds shape:", current_prompt_embeds.shape)
            # print("Pooled_prompt_embeds shape:", current_pooled_prompt_embeds.shape)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = pipeline.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=current_pooled_prompt_embeds,
                encoder_hidden_states=current_prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                joint_attention_kwargs=pipeline.joint_attention_kwargs,
                return_dict=False,
            )[0]

            noise_pred = noise_pred.to(prompt_embeds.dtype)
            latents_dtype = latents.dtype

            latents, log_prob, prev_latents_mean, std_dev_t = denoising_sde_step_with_logprob(
                scheduler=pipeline.scheduler,
                model_output=noise_pred.float(),
                timestep=t.unsqueeze(0).repeat(latents.shape[0]),
                sample=latents.float(),
                noise_level=current_noise_level,
                prev_sample=None,
            )
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            if layout is not None and i < pipeline.scheduler.merge_step:
                # Reconstruct full latents and compute the mean log_prob if use dividing
                latents = latents.view(batch_size, layout[0], layout[1], -1, latents.shape[2]) # (B, rows, cols, sub_seq_len, C)
                latents = merge_latents(latents, height, width, sub_height, sub_width) # (B, seq_len, C)
                # TODO: for subfig generation, should we compute the log_prob of the full image , or the mean of all subfig log_probs?
                log_prob = log_prob.view(batch_size, layout[0], layout[1]) # (B, rows, cols)
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim))) # (B,)

            if current_noise_level > 0:
                all_latents.append(latents)
                all_log_probs.append(log_prob) # mean along all but batch dimension.
                all_timestep_indices.append(i)
    
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

    return images, all_latents, all_log_probs, all_timestep_indices
