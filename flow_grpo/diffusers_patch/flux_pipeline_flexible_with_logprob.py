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
from ..scheduler import FlowMatchSlidingWindowScheduler

def denoising_sde_step_with_logprob(
    scheduler: FlowMatchSlidingWindowScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, list[float], torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: Union[int, float, list[float], torch.FloatTensor] = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Predict the sample from the previous timestep by **reversing** the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity). Specially, when noise_level is zero, the process becomes deterministic.

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float` | `list[float]` | `torch.FloatTensor`):
            The current discrete timestep(s) in the diffusion chain, with batch dimension.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        noise_level (`int` | `float` | `list[float]` | `torch.FloatTensor`, *optional*, defaults to 0.7):
            The noise level parameter, can be different for each sample in the batch. This parameter controls the standard deviation of the noise added to the denoised sample.
        prev_sample (`torch.FloatTensor`):
            The next insance of the sample. If given, calculate the log_prob using given `prev_sample` as predicted value.
        generator (`torch.Generator`, *optional*):
            A random number generator for SDE solving. If not given, a random generator will be used.
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    if isinstance(timestep, float) or isinstance(timestep, int):
        # Convert single value to a tensor with shape (batch_size,)
        timestep = [timestep] * sample.shape[0]

    # Convert noise_level to a tensor with shape (batch_size, 1, 1)
    if isinstance(noise_level, float) or isinstance(noise_level, int):
        noise_level = torch.tensor([noise_level], device=sample.device, dtype=sample.dtype).expand(sample.shape[0])
    elif isinstance(noise_level, list):
        noise_level = torch.tensor(noise_level, device=sample.device, dtype=sample.dtype)
    elif isinstance(noise_level, torch.Tensor):
        noise_level = noise_level.to(device=sample.device, dtype=sample.dtype)

    step_index = [scheduler.index_for_timestep(t) for t in timestep]
    prev_step_index = [step + 1 for step in step_index]
    # sigmas is a decreasing sequence from 1 to 0, sigma=1 means pure noise, sigma=0 means pure data
    # sigma here has shape (batch_size, 1, 1)
    sigma = scheduler.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = scheduler.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_max = scheduler.sigmas[1].item()
    dt = sigma_prev - sigma # dt is negative, (batch_size, 1, 1)

    noise_level = noise_level.view(-1, *([1] * (len(sample.shape) - 1)))

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level # (batch_size, 1, 1)
    
    # our sde
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

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
        - torch.log(std_dev_t * torch.sqrt(-1 * dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
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
    timestep_index = scheduler.left_boundary + j # timestep index in the scheduler.timesteps

    batch_size = latents.shape[0]
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    height = sample["heights"][j]
    width = sample["widths"][j]
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
        noise_level=config.sample.noise_level,
        prev_sample=sample["next_latents"][:, j].float(),
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t


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
) -> Tuple[
        torch.FloatTensor,
        List[torch.FloatTensor],
        List[torch.FloatTensor]
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
    pipeline.scheduler.set_begin_index(0)
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            pipeline._current_timestep = t
            if i == pipeline.scheduler.left_boundary:
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

            if pipeline.scheduler.left_boundary <= i < pipeline.scheduler.right_boundary:
                all_latents.append(latents)
                all_log_probs.append(log_prob)
    
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

    return images, all_latents, all_log_probs
