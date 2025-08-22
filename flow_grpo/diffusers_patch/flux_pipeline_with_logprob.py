# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py

from typing import Any, Dict, List, Optional, Union, Callable
import torch
import numpy as np
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from .sd3_sde_with_logprob import denoising_step_with_logprob

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

@torch.no_grad()
def pipeline_with_logprob(
    pipeline,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt: Union[str, List[str]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
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
):
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
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    # 4. Prepare latent variables
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    latents, latent_image_ids = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
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
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)
    pipeline._num_timesteps = len(timesteps)

    # handle guidance
    if pipeline.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None
    
    # 6. Prepare image embeddings
    all_latents = []
    all_log_probs = []

    # 7. Denoising loop
    pipeline.scheduler.set_begin_index(0)
    scheduler = pipeline.scheduler
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            pipeline._current_timestep = t
            if i == scheduler.left_boundary:
                all_latents.append(latents)

            # Get noise_level. If not given in the arguments, use the sliding window scheduler's method to retrieve it.
            current_noise_level = noise_level if noise_level is not None else pipeline.scheduler.get_noise_level_for_timestep(t)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = pipeline.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=pipeline.joint_attention_kwargs,
                return_dict=False,
            )[0]

            noise_pred = noise_pred.to(prompt_embeds.dtype)
            latents_dtype = latents.dtype

            latents, log_prob, prev_latents_mean, std_dev_t = denoising_step_with_logprob(
                pipeline.scheduler,
                noise_pred.float(),
                t.unsqueeze(0).repeat(latents.shape[0]),
                latents.float(),
                noise_level=current_noise_level,
                prev_sample=None,
                # generator=noise_gens # Add different generator for each step, a solution is to use given generator to generate new random generators.
            )
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            if pipeline.scheduler.left_boundary <= t < pipeline.scheduler.right_boundary:
                all_latents.append(latents)
                all_log_probs.append(log_prob)
    
            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

    latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    latents = latents.to(dtype=pipeline.vae.dtype)
    image = pipeline.vae.decode(latents, return_dict=False)[0]
    image = pipeline.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    pipeline.maybe_free_model_hooks()

    return image, all_latents, latent_image_ids, text_ids, all_log_probs
