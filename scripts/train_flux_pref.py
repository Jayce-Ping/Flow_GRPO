from argparse import Namespace
import contextlib
import datetime
import hashlib
import json
import numpy as np
import os
import random
import signal
import sys
import tempfile
import time
import torch
import tqdm
from typing import List, Tuple, Any, Optional
import shutil

from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from collections import defaultdict
from concurrent import futures
from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from functools import partial
from ml_collections import config_flags
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler

from flow_grpo.utils import tensor_to_pil_image, numpy_to_pil_image, numpy_list_to_pil_image, tensor_list_to_pil_image, gather_tensor_list
from flow_grpo.rewards.rewards import multi_score
from flow_grpo.rewards.pref_scorer import pref_score
from flow_grpo.diffusers_patch.flux_pipeline_flexible_with_logprob import calculate_shift, pipeline_with_logprob, denoising_sde_step_with_logprob, compute_log_prob
from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.datasets.prompt_dataset import TextPromptDataset, GenevalPromptDataset
from flow_grpo.datasets.sampler import DistributedKRepeatSampler
from flow_grpo.scheduler import FlowMatchSlidingWindowScheduler
from flow_grpo.memory_tracker import MemoryProfiler

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds

def create_generator(prompts : List[str], base_seed : int) -> List[torch.Generator]:
    generators = []
    for batch_pos, prompt in enumerate(prompts):
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


@torch.no_grad()
def eval(pipeline : FluxPipeline,
         test_dataloader : DataLoader,
         text_encoders,
         tokenizers,
         config : Namespace,
         accelerator,
         logging_platform,
         global_step,
         reward_fn,
         executor,
         autocast,
         ema,
         transformer_trainable_parameters,
         memory_profiler : Optional[MemoryProfiler] = None,
         log_sample_num : int = 90 # 108 as max in wandb/swanlab
    ):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    
    log_data = {
        'images': [],
        'prompts': [],
        'rewards': defaultdict(list)
    }
    if memory_profiler is not None:
        memory_profiler.snapshot("before_eval")

    for batch_idx, test_batch in enumerate(tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        )):
        if memory_profiler is not None:
            memory_profiler.snapshot(f"eval_batch_{batch_idx}_start")

        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=config.max_sequence_length,
            device=accelerator.device
        )
        heights = [prompt_meta.get('height', config.resolution) for prompt_meta in prompt_metadata]
        widths = [prompt_meta.get('width', config.resolution) for prompt_meta in prompt_metadata]
        if not all(h == heights[0] for h in heights) or not all(w == widths[0] for w in widths):
            # Split the batch if there are different sizes
            images = []
            for i in tqdm(
                range(len(prompts)),
                desc="Eval: per sample",
                leave=False,
                position=1,
                disable=not accelerator.is_local_main_process,
            ):
                prompt = [prompts[i]]
                prompt_meta = [prompt_metadata[i]]
                height = heights[i]
                width = widths[i]
                with autocast():
                    imgs, _, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds[i].unsqueeze(0),
                        pooled_prompt_embeds=pooled_prompt_embeds[i].unsqueeze(0),
                        num_inference_steps=config.test.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=height,
                        width=width,
                        noise_level=0,
                    )

                images.append(imgs.squeeze(0))  # (C, H, W)
        else:
            # Batch inference if all sizes are the same
            with autocast():
                images, _, _, = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=config.test.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=heights[0],
                    width=widths[0],
                    noise_level=0,
                )
                images = list(images.unbind(0)) # List[torch.Tensor(C, H, W)]
        # reward_fn accepts torch.Tensor (B, C, H, W) or List[torch.Tensor(C, H, W)]
        future = executor.submit(reward_fn, images, prompts, prompt_metadata)
        # yield to to make sure reward computation starts
        time.sleep(0)
        # all_futures.append(future)
        rewards, reward_metadata = future.result()
    
        # -------------------------------Collect log data--------------------------------
        if len(log_data["prompts"]) < log_sample_num // accelerator.num_processes:
            log_data['images'].extend(images)
            log_data['prompts'].extend(prompts)
            for key, value in rewards.items():
                if key not in log_data['rewards']:
                    log_data['rewards'][key] = []
                
                log_data['rewards'][key].extend(value)
        
        # log memory after reward computation
        if memory_profiler is not None:
            memory_profiler.snapshot(f"eval_batch_{i}_end")

    if memory_profiler is not None:
        memory_profiler.snapshot("after_eval_before_gather_log_data")
    # ---------------------------Gather all Log data, with prompt-image-reward tuples--------------------------
    # 1. Gather all rewards and report average
    gathered_rewards = {}
    for key, value in log_data['rewards'].items():
        gathered_rewards[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    if accelerator.is_main_process:
        # Report detailed rewards values
        for key, value in gathered_rewards.items():
            print(key, np.mean(value))

        # Log eval metrics
        logging_platform.log(
            {f"eval/{key}": np.mean(value) for key, value in gathered_rewards.items()},
            step=global_step
        )

    # gathered_rewards = {'r1': [1,2,3], 'r2': [4,5,6]}
    # ->
    # gathered_rewards = [{'r1':1, 'r2':4}, {'r1':2, 'r2':5}, {'r1':3, 'r2':6}]
    gathered_rewards = [
        dict(zip(gathered_rewards.keys(), value))
        for value in zip(*gathered_rewards.values())
    ]

    if memory_profiler is not None:
        memory_profiler.snapshot("after_gather_rewards")

    # 2. Encode prompt to tensors for gpu communication
    prompt_ids = tokenizers[1](
        log_data['prompts'],
        padding="max_length",
        max_length=config.max_sequence_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    gathered_prompt_ids = accelerator.gather(prompt_ids).cpu().numpy()
    gathered_prompts = tokenizers[1].batch_decode(
        gathered_prompt_ids, skip_special_tokens=True
    )

    if memory_profiler is not None:
        memory_profiler.snapshot("after_gather_prompts")

    # 3. Gather all images
    use_jpg_compression = True
    # Approach : by saving them in a temp dir
    if use_jpg_compression:
        # This approach saves images as JPG files in a temporary directory
        # Since uploading images with jpg is faster, if we need to do it anyway.
        temp_dir = os.path.join(config.save_dir, 'temp_eval_images')
        os.makedirs(temp_dir, exist_ok=True)
        offset = accelerator.process_index * len(log_data['images'])
        for i,img in enumerate(log_data['images']):
            # Save image to temp dir
            pil_img = tensor_to_pil_image(img)[0]
            pil_img.save(os.path.join(temp_dir, f"{offset + i}.jpg"))
        accelerator.wait_for_everyone()
        # The order of images here should be guaranteed by the name of images
        # NOTE: it provides gathered_images as a list of file paths
        gathered_images = [os.path.join(temp_dir, f) for f in sorted(os.listdir(temp_dir), key=lambda x: int(x.split('.')[0]))]
    else:
        # Approach: flatten and gather, then reshape
        gathered_images = gather_tensor_list(accelerator, log_data['images'], device="cpu")
        gathered_images = tensor_list_to_pil_image(gathered_images) # List[PIL.Image]

    if memory_profiler is not None:
        memory_profiler.snapshot("after_gather_images")

    if accelerator.is_main_process:
         # Use a fixed generator to log same indices everytime for comparison
        gen = torch.Generator().manual_seed(0)
        # Sample `log_sample_num` data for logging, 'None' for all data.
        if log_sample_num is None:
            sample_indices = list(range(len(gathered_prompts)))
        else:
            sample_indices = torch.randperm(len(gathered_prompts), generator=gen)[:log_sample_num]

        sampled_images = [gathered_images[i] for i in sample_indices]
        sampled_prompts = [gathered_prompts[i] for i in sample_indices]
        sampled_rewards = [gathered_rewards[i] for i in sample_indices]

        logging_platform.log(
            {
                "eval_images": [
                    logging_platform.Image(
                        image,
                        caption=", ".join(f"{k}: {v:.2f}" for k, v in reward.items()) + f" | {prompt}",
                    )
                    for idx, (image, prompt, reward) in enumerate(zip(sampled_images, sampled_prompts, sampled_rewards))
                ]
            },
            step=global_step,
        )
        # Clean up temp dir
        if use_jpg_compression:
            shutil.rmtree(temp_dir)

    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

    # Log memory after eval
    if memory_profiler is not None:
        memory_profiler.snapshot("after_eval")

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def load_pipeline(config : Namespace, accelerator : Accelerator):
    # -------------------------------Load models-----------------------------------
    # load scheduler, tokenizer and models.
    pipeline = FluxPipeline.from_pretrained(
        config.pretrained.model,
        low_cpu_mem_usage=False
    )

    if config.sample.use_sliding_window:
        scheduler = FlowMatchSlidingWindowScheduler(
            noise_level=config.sample.noise_level,
            window_size=config.sample.window_size,
            left_boundary=config.sample.left_boundary,
            **pipeline.scheduler.config.__dict__,
        )
    else:
        scheduler = FlowMatchSlidingWindowScheduler(
            noise_level=config.sample.noise_level,
        )

    # Overwrite the original scheduler
    pipeline.scheduler = scheduler

    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    
    pipeline.transformer.to(accelerator.device)

    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)


    return pipeline, text_encoders, tokenizers

def setup_wandb_log(accelerator, config):
    """
        Initialize wandb training log
    """
    import wandb
    if config.resume_from_id is not None:
        project_name = config.project_name
        run_id = config.resume_from_id
        # Get history
        api_run = wandb.Api().run(f"{project_name}/{run_id}")
        history = api_run.history()
        if not history.empty:
            if config.resume_from_step is None:
                config.resume_from_step = int(history['_step'].iloc[-1])
            if config.resume_from_epoch is None:
                config.resume_from_epoch = config.resume_from_step // 2
            logger.info(f"Auto-resuming from step {config.resume_from_step}, epoch {config.resume_from_epoch}")
        else:
            logger.info("No previous history found, starting from beginning")
            config.resume_from_step = 0
            config.resume_from_epoch = 0

        if accelerator.is_main_process:
            run = wandb.init(
                project=config.project_name,
                config=config.to_dict(),
                id=run_id,
                resume='must'
            )
        else:
            run = None
    else:
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        if not config.run_name:
            config.run_name = unique_id
        else:
            config.run_name += "_" + unique_id
        if accelerator.is_main_process:
            run = wandb.init(
                project=config.project_name,
                config=config.to_dict()
            )
        else:
            run = None

    return run, wandb

def setup_swanlab_log(accelerator, config):
    """
        Initialize swanlab training log
    """
    import swanlab
    if config.resume_from_id:
        project_name = config.project_name
        run_id = config.resume_from_id
        # Get history
        api = swanlab.OpenApi()
        run_summary = api.get_summary(project=project_name, exp_id=run_id)
        if config.resume_from_step is None:
            config.resume_from_step = run_summary.data['epoch']['max']['step']
        if config.resume_from_epoch is None:
            config.resume_from_epoch = run_summary.data['epoch']['max']['value']
        logger.info(f"Auto-resuming from step {config.resume_from_step}, epoch {config.resume_from_epoch}")

        if accelerator.is_main_process:
            run = swanlab.init(
                project=project_name,
                config=config.to_dict(),
                resume=True,
                id=run_id
            )
        else:
            run = None
    else:
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        if not config.run_name:
            config.run_name = unique_id
        else:
            config.run_name += "_" + unique_id

        if accelerator.is_main_process:
            run = swanlab.init(
                project=config.project_name,
                config=config.to_dict()
            )
        else:
            run = None

    return run, swanlab

def set_online_log(accelerator, config):
    """
        Initialize logging with platform
    """
    if config.logging_platform == 'wandb':
        run, logging_platform = setup_wandb_log(accelerator, config)
    elif config.logging_platform == 'swanlab':
        run, logging_platform = setup_swanlab_log(accelerator, config)
    else:
        raise ValueError(f"Unsupported logging platform: {config.logging_platform}")
    
    return run, logging_platform

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    # Flexible training only supports batch size 1, so
    # update gradient_accumulation_steps, and update train.batch_size to 1 later for logger info
    if config.enable_flexible_size:
        gradient_accumulation_steps = config.train.gradient_accumulation_steps * config.train.batch_size
    else:
        gradient_accumulation_steps = config.train.gradient_accumulation_steps

    # number of timesteps within each trajectory to train on
    if config.sample.use_sliding_window:
        num_train_timesteps = config.sample.window_size 
    else:
        num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=gradient_accumulation_steps * num_train_timesteps,
    )
    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    if config.enable_flexible_size and config.train.batch_size != 1:
        # Print a warning message and override config
        logger.info(
            "Only batch size 1 is supported for flexible size training: "
            f"Overriding config.train.gradient_accumulation_steps by multiplying it with config.train.batch_size {config.train.gradient_accumulation_steps}*{config.train.batch_size}={gradient_accumulation_steps}"
            f" and setting config.train.batch_size to 1")
        
        config.train.batch_size = 1
        config.train.gradient_accumulation_steps = gradient_accumulation_steps
    else:
        # config.train.batch_size should divide config.sample.batch_size * config.num_batches_per_epoch
        assert (config.sample.batch_size * config.sample.num_batches_per_epoch) % config.train.batch_size == 0, \
            f"config.train.batch_size {config.train.batch_size} should divide config.sample.batch_size {config.sample.batch_size} * config.num_batches_per_epoch {config.sample.num_batches_per_epoch}"

    # -------------------------------------------------Set up online log-----------------------------------
    if not config.project_name:
        config.project_name = 'FlowGRPO-Flux'

    run, logging_platform = set_online_log(accelerator, config)

    def safe_exit(sig, frame):
        print("Received signal to terminate.")
        if accelerator.is_main_process:
            logging_platform.finish()
        
        sys.exit(0)

    signal.signal(signal.SIGINT, safe_exit)
    
    logger.info(f"\n{config}")

    # -----------------------------------------------Set up memory profiler-----------------------------------
    if config.enable_mem_log:
        # Initialize memory profiler
        time_stamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        meme_log_file = f'memory_{time_stamp}.log'
        # clean up old log file
        if accelerator.is_main_process and os.path.exists(meme_log_file):
            os.remove(meme_log_file)
        memory_profiler = MemoryProfiler(accelerator, enable_tensor_accumulation=True, log_file=meme_log_file)

    # --------------------------------------Load pipeline----------------------------------
    pipeline, text_encoders, tokenizers = load_pipeline(config, accelerator)
    transformer = pipeline.transformer

    if config.enable_mem_log:
        # Register model to profiler
        memory_profiler.register_model(transformer, "transformer")
        memory_profiler.snapshot("after_model_loading")
    
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    if config.enable_mem_log:
        memory_profiler.track_optimizer(optimizer)
        memory_profiler.snapshot("after_optimizer_init")

    # ---------------------------------------Data---------------------------------------
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')
        collate_fn = TextPromptDataset.collate_fn
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')
        collate_fn = GenevalPromptDataset.collate_fn
    else:
        raise NotImplementedError("Specify `prompt_fn` in ['general_ocr', 'geneval']")

    # Create an infinite-loop DataLoader
    train_sampler = DistributedKRepeatSampler( 
        dataset=train_dataset,
        batch_size=config.sample.batch_size,
        k=config.sample.num_image_per_prompt,
        m=config.sample.unique_sample_num_per_epoch,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=config.seed
    )

    assert config.sample.num_batches_per_epoch == train_sampler.num_batches_per_epoch, \
        f"""
config.sample.num_batches_per_epoch={config.sample.num_batches_per_epoch},
train_sampler.num_batches_per_epoch={train_sampler.num_batches_per_epoch},
These two numbers should be equal
        """

    # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=collate_fn
    )

    # Create a regular DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.test.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=8,
    )

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # Initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std, config.sample.use_history)

    # For some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # for deepspeed zero
    if accelerator.state.deepspeed_plugin:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.sample.batch_size

    # Prepare everything with our `accelerator`.
    transformer, optimizer, test_dataloader = accelerator.prepare(transformer, optimizer, test_dataloader)
    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    if config.enable_mem_log:
        memory_profiler.snapshot("after_accelerator_prepare")
    # -----------------------------------------Reward fn-----------------------------------------
    # prepare prompt and reward fn
    if accelerator.is_main_process:
        print(f"Reward dict: {config.reward_fn}")
    eval_reward_fn = multi_score(accelerator.device, config.reward_fn, config.aggregate_fn)
    pref_reward_fn = pref_score()

    if config.enable_mem_log:
        memory_profiler.snapshot("after_loading_reward_fn")
    # ------------------------------------------- Train!------------------------------------------
    samples_per_epoch = (
        config.sample.batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.batch_size >= config.train.batch_size
    # assert config.sample.batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from_id:
        global_step = config.resume_from_step
        epoch = global_step // 2
    else:
        global_step = 0
        epoch = 0
    

    while True:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if config.eval_freq > 0 and epoch % config.eval_freq == 0:
            if config.enable_mem_log:
                memory_profiler.snapshot(f"epoch_{epoch}_before_eval")
            eval(
                pipeline,
                test_dataloader,
                text_encoders,
                tokenizers,
                config,
                accelerator,
                logging_platform,
                global_step,
                eval_reward_fn, executor,
                autocast,
                ema,
                transformer_trainable_parameters,
                memory_profiler=memory_profiler if config.enable_mem_log else None,
            )
            if config.enable_mem_log:
                memory_profiler.snapshot(f"epoch_{epoch}_after_eval")
    
        if config.save_freq > 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        train_sampler.set_epoch(epoch)
        train_iter = iter(train_dataloader)

        samples = []
        for i in tqdm(
            range(train_sampler.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            prompts, prompt_metadata = next(train_iter)
            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts,
                text_encoders,
                tokenizers,
                max_sequence_length=config.max_sequence_length,
                device=accelerator.device
            )
            prompt_ids = tokenizers[1](
                prompts,
                padding="max_length",
                max_length=config.max_sequence_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # Get heights and widths
            heights = [prompt_meta.get('height', config.resolution) for prompt_meta in prompt_metadata]
            widths = [prompt_meta.get('width', config.resolution) for prompt_meta in prompt_metadata]
            # Fixed size training requires all heights and widths in the batch to be the same
            if not config.enable_flexible_size:
                assert all(h == heights[0] for h in heights) and all(w == widths[0] for w in widths), \
                    f"When config.enable_flexible_size is False, all heights and widths in the batch must be the same, but got heights {heights} and widths {widths}"

            # sample
            if config.sample.same_latent:
                # Same seed for same prompt
                generators = create_generator(prompts, base_seed=epoch)
            else:
                # Different initial latent seed
                generators = None

            # If all heights and widths are the same, we can batch them together
            if all(h == heights[0] for h in heights) and all(w == widths[0] for w in widths):
                with autocast():
                    with torch.no_grad():
                        images, all_latents, all_log_probs = pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            height=heights[0],
                            width=widths[0],
                            generator=generators
                    )
                    # images: (batch_size, C, H, W) -> List[Tensor(C, H, W)] with length batch_size
                    # all_latents: List[Tensor(batch_size C, H, W)] with length windowsize+1 -> List[Tensor(window_size + 1, C, H, W)] with length batch_size
                    # all_log_probs: List[Tensor(batch_size)] with length window_size -> List[Tensor(window_size) with length batch_size
                    images = list(images.unbind(0)) # List[Tensor(C, H, W)] with length batch_size
                    all_latents = torch.stack(all_latents, dim=1) # (batch_size, window_size + 1, C, H, W)
                    all_latents = list(all_latents.unbind(0)) # List[Tensor(window_size + 1, C, H, W)] with length batch_size
                    all_log_probs = torch.stack(all_log_probs, dim=1) # (batch_size, window_size)
                    all_log_probs = list(all_log_probs.unbind(0)) # List[Tensor(window_size)] with length batch_size
            else:
                # Different sizes, have to do one by one
                images = []
                all_latents = []
                all_log_probs = []
                for index in range(len(prompts)):
                    with autocast():
                        with torch.no_grad():
                            this_image, this_all_latents, this_all_log_probs = pipeline_with_logprob(
                                pipeline,
                                prompt_embeds=prompt_embeds[index].unsqueeze(0),
                                pooled_prompt_embeds=pooled_prompt_embeds[index].unsqueeze(0),
                                num_inference_steps=config.sample.num_steps,
                                guidance_scale=config.sample.guidance_scale,
                                output_type="pt",
                                height=heights[index],
                                width=widths[index],
                                generator=generators[index] if generators is not None else None
                        )
                    images.append(this_image.squeeze(0))  # add (C, H, W)
                    all_latents.append(torch.stack(this_all_latents, dim=1).squeeze(0))  # add (window_size + 1, C, H, W)
                    all_log_probs.append(torch.stack(this_all_log_probs, dim=1).squeeze(0))  # add (window_size, )

                # images: List[Tensor(C, H, W)] with length batch_size
                # all_latents: List[Tensor(window_size + 1, C, H, W)] with length batch_size
                # all_log_probs: List[Tensor(window_size)] with length batch_size
            
            samples.extend(
                [
                    {
                        'height': heights[index],
                        'width': widths[index],
                        'image': images[index],
                        'prompt_ids': prompt_ids[index].unsqueeze(0), # Keep batch dimension as 1
                        'prompt_embeds': prompt_embeds[index].unsqueeze(0),
                        'pooled_prompt_embeds': pooled_prompt_embeds[index].unsqueeze(0),
                        'latents': all_latents[index][:-1].unsqueeze(0),
                        'next_latents': all_latents[index][1:].unsqueeze(0),
                        'log_probs': all_log_probs[index].unsqueeze(0),
                    }
                    for index in range(len(prompts))
                ]
            )

            if config.enable_mem_log:
                memory_profiler.track_samples(samples, f"sampling")
                memory_profiler.snapshot(f"epoch_{epoch}_after_sampling_batch_{i}")

        # ---------------------------Compute rewards---------------------------
        # Since the Pref-reward is computed within the whole group, we need to group images with the same prompt together

        # Image communication
        approach = 2
        if approach == 1:
            # Approach 1: save in a temp dir instead
            # Slower but saves GPU memory
            temp_dir = os.path.join(config.save_dir, 'temp_train_images')
            os.makedirs(temp_dir, exist_ok=True)
            for i, s in enumerate(samples):
                img = s["image"]
                img_id = f"{accelerator.process_index * len(samples) + i}.jpg"
                pil_img = numpy_to_pil_image(img)[0]
                pil_img.save(os.path.join(temp_dir, img_id))
            
            accelerator.wait_for_everyone()
            gathered_images = [Image.open(os.path.join(temp_dir, f)) for f in sorted(os.listdir(temp_dir), key=lambda x: int(x.split('.')[0]))]
        else:
            # Approach 2: Flattern, gather and reshape
            # Get shapes and flatten lengths for each image
            local_shapes = torch.tensor([list(s['image'].shape) for s in samples], device=accelerator.device, dtype=torch.long) # (B, 3)
            # Gather shapes and lengths
            gathered_shapes = accelerator.gather(local_shapes).cpu() # (sum_B, 3)
            # Flatten and gather images
            flat_images = torch.cat([s['image'].flatten() for s in samples], dim=0) # (sum_local_length,)
            gathered_flat_images = accelerator.gather(flat_images).cpu() # (sum_global_length,)

            gathered_images = []
            offset = 0
            for shape in gathered_shapes:
                C, H, W = map(int, shape.tolist())
                length = C * H * W
                if length > 0:
                    img = gathered_flat_images[offset:offset+length].reshape(C, H, W)
                    gathered_images.append(img)
                    offset += length

            gathered_images = tensor_list_to_pil_image(gathered_images) # List[PIL.Image]

        # Gather all prompts
        gathered_prompt_ids = accelerator.gather(torch.cat([s["prompt_ids"] for s in samples], dim=0))
        gathered_prompts = tokenizers[1].batch_decode(gathered_prompt_ids, skip_special_tokens=True)
        gathered_pref_rewards = np.zeros((len(gathered_prompts),))  # placeholder to fill in later
        prompt_to_pos = defaultdict(list)
        for i, prompt in enumerate(gathered_prompts):
            prompt_to_pos[prompt].append(i)
        
        # Split into num_processes chunks to distribute the reward computation
        dist_prompts = list(prompt_to_pos.keys())[accelerator.process_index::accelerator.num_processes]
        for prompt in dist_prompts:
            images = [gathered_images[i] for i in prompt_to_pos[prompt]]
            group_size = len(images)
            # compute reward for each prompt
            rewards, _ = executor.submit(pref_reward_fn, images, [prompt]*group_size, [{}]*group_size).result()
            gathered_pref_rewards[prompt_to_pos[prompt]] = rewards
        
        # Gather all pref rewards
        gathered_pref_rewards = accelerator.gather(torch.as_tensor(gathered_pref_rewards, device=accelerator.device)).cpu().numpy()
        gathered_pref_rewards = gathered_pref_rewards.reshape(accelerator.num_processes, -1).sum(axis=0)

        gathered_rewards = {
            'avg': np.array(gathered_pref_rewards),
            'pref_score': np.array(gathered_pref_rewards),
        }
        # log rewards and images
        if accelerator.is_main_process:
            print(f"Epoch {epoch} rewards: ")
            for key, value in gathered_rewards.items():
                print(f"  {key}: {value.mean():.4f} ± {value.std():.4f}")
            logging_platform.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items()},
                },
                step=global_step,
            )

        # ----------------------------------Compute advantages----------------------------------
        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            advantages = stat_tracker.update(gathered_prompts, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(gathered_prompts))
                print("len unique prompts", len(set(gathered_prompts)))

                (
                    avg_group_size,
                    trained_prompt_num,
                    avg_group_std,
                    global_std,
                    zero_std_ratio
                ) = stat_tracker.get_stats()

                if accelerator.is_main_process:
                    logging_platform.log(
                        {
                            "avg_group_size": avg_group_size,
                            "trained_prompt_num": trained_prompt_num,
                            "avg_group_std": avg_group_std,
                            "global_std": global_std,
                            "zero_std_ratio": zero_std_ratio,
                        },
                        step=global_step,
                    )
            # !!! Notice here, after every advantage calculation, the tracker is cleared so that no history is saved.
            # So comment the following clear code if `config.sample.use_history=True` is set
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        advantages = (
            advantages.reshape(accelerator.num_processes, -1, *advantages.shape[1:])[accelerator.process_index]
            .to(accelerator.device)
        )
        for i, sample in enumerate(samples):
            sample['advantages'] = advantages[i]

        if accelerator.is_local_main_process:
            print("len samples", len(samples))
            print("advantages has shape", advantages.shape)

        # clean up to save memory
        del gathered_rewards
        for sample in samples:
            del sample["prompt_ids"]
            del sample["image"]
        
        # clean up temp dir
        if accelerator.is_main_process:
            shutil.rmtree(temp_dir)


        #################### TRAINING ####################
        if config.enable_mem_log:
            memory_profiler.snapshot(f"epoch_{epoch}_before_training")

        total_batch_size = len(samples) # = config.train.batch_size * config.train.num_batches_per_epoch

        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples
            perm = torch.randperm(total_batch_size)
            samples = [samples[i] for i in perm]

            if config.enable_flexible_size:
                assert config.train.batch_size == 1, "Only batch size 1 is supported for flexible size training"
            else:
                # sample:{
                # 'height': int,
                # 'width': int,
                # 'prompt_embeds': Tensor(1, L, D),
                # 'pooled_prompt_embeds': Tensor(1, D),
                # 'latents': Tensor(1, window_size + 1, C, H
                # 'next_latents': Tensor(1, window_size + 1, C, H, W),
                # 'log_probs': Tensor(1, window_size),
                # 'advantages': Tensor(1, 1),
                # }
                keys = samples[0].keys()
                samples = [samples[i:i+config.train.batch_size] for i in range(0, total_batch_size, config.train.batch_size)]
                samples = [
                    {
                         # Catenate along batch dimension if the entry is Tensor
                        k: torch.cat([s[k] for s in batch], dim=0)
                        if isinstance(batch[0][k], torch.Tensor)
                        else batch[0][k] # for other type -  they should be the same within the batch
                        for k in keys
                    }
                    for batch in samples
                ]

            pipeline.transformer.train()
            info = defaultdict(list)

            for i, sample in tqdm(
                list(enumerate(samples)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    if config.enable_mem_log and i % 10 == 0:
                        memory_profiler.snapshot(f"epoch_{epoch}_step_{i}_timestep_{j}_before_forward")

                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, config)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        # Disable adapter to get the original reference model parameters.
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, config)

                        # grpo logic
                        advantages = torch.clamp(
                            sample["advantages"],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )

                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        # print("ratio", ratio)
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (
                                    ratio - 1.0 > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (
                                    1.0 - ratio > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        # Track training tensors
                        training_tensors = {
                            "prev_sample": prev_sample,
                            "log_prob": log_prob,
                            "advantages": advantages,
                            "ratio": ratio,
                            "loss": loss,
                        }
                        if config.enable_mem_log:
                            memory_profiler.track_tensors(training_tensors, "training")
                            if i % 10 == 0:
                                memory_profiler.snapshot(f"epoch_{epoch}_step_{i}_timestep_{j}_before_backward")

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                        if config.enable_mem_log and i % 10 == 0:
                            memory_profiler.snapshot(f"epoch_{epoch}_step_{i}_timestep_{j}_after_backward")

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            logging_platform.log(info, step=global_step)

                        if config.enable_mem_log:
                            memory_profiler.snapshot(f"epoch_{epoch}_step_{i}_after_optimization")
                            memory_profiler.print_full_report(f"epoch_{epoch}_step_{i}")

                        global_step += 1
                        info = defaultdict(list)

                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients

        if config.enable_mem_log:
            memory_profiler.cleanup_and_snapshot(f"epoch_{epoch}_end")
            # Clear tensor accumulation info in profiler to save memory
            memory_profiler.tensor_tracker.clear_stats()

        epoch += 1
        
if __name__ == "__main__":
    app.run(main)

