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
from functools import partial
from ml_collections import config_flags
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler

from flow_grpo.rewards.rewards import multi_score
from flow_grpo.diffusers_patch.flux_pipeline_with_logprob import pipeline_with_logprob, compute_log_prob
from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.datasets.prompt_dataset import TextPromptDataset, GenevalPromptDataset
from flow_grpo.datasets.sampler import DistributedKRepeatSampler
from flow_grpo.scheduler import FlowMatchSlidingWindowScheduler
from flow_grpo.debug_utils import MemoryProfiler

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
         log_sample_num : int = 90 # 108 as max in wandb/swanlab
    ):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    log_data = {
        'images': [],
        'prompts': [],
        'rewards': defaultdict(list)
    }
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=config.max_sequence_length,
            device=accelerator.device
        )
        with autocast():
            images, _, _, _, _ = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=config.eval_numsteps,
                guidance_scale=config.sample.guidance_scale,
                output_type="pt",
                height=config.resolution,
                width=config.resolution, 
                noise_level=0,
            )
        future = executor.submit(reward_fn, images, prompts, prompt_metadata)
        # yield to to make sure reward computation starts
        time.sleep(0)
        # all_futures.append(future)
        rewards, reward_metadata = future.result()
        
        # -------------------------------Collect log data--------------------------------
        if len(log_data["prompts"]) < log_sample_num // accelerator.num_processes:
            log_data['images'].extend([img.cpu().numpy() for img in images])
            log_data['prompts'].extend(prompts)
            for key, value in rewards.items():
                if key not in log_data['rewards']:
                    log_data['rewards'][key] = []
                
                log_data['rewards'][key].extend(value)


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

    # 3. Gather all images
    # Approach : by saving them in a temp dir
    temp_dir = os.path.join(config.save_dir, 'temp_eval_images')
    os.makedirs(temp_dir, exist_ok=True)
    for i,img in enumerate(log_data['images']):
        # Save image to temp dir
        img_id = f"{accelerator.process_index * len(log_data['images']) + i}.jpg"
        pil = Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8))
        pil.save(os.path.join(temp_dir, img_id))
    
    accelerator.wait_for_everyone()
    # The order of images here should be guaranteed by the name of images
    # NOTE: it provides gathered_images as a list of file paths
    gathered_images = [os.path.join(temp_dir, f) for f in sorted(os.listdir(temp_dir), key=lambda x: int(x.split('.')[0]))]

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
        shutil.rmtree(temp_dir)

    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

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
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

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

    # -------------------------------------------------Set up memory profiler-----------------------------------
    if config.enable_mem_log:
        # Initialize memory profiler
        time_stamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        meme_log_file = f'memory_{time_stamp}.log'
        # clean up old log file
        if accelerator.is_main_process and os.path.exists(meme_log_file):
            os.remove(meme_log_file)

        memory_profiler = MemoryProfiler(accelerator=accelerator, log_file=meme_log_file)

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
        collate_fn=collate_fn,
        # persistent_workers=True
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
    reward_fn = multi_score(accelerator.device, config.reward_fn, config.aggregate_fn)
    
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
                reward_fn, executor,
                autocast,
                ema,
                transformer_trainable_parameters
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

            # sample
            if config.sample.same_latent:
                # Same seed for same prompt
                generator = create_generator(prompts, base_seed=epoch)
            else:
                # Different initial latent seed
                generator = None
            with autocast():
                with torch.no_grad():
                    images, all_latents, image_ids, text_ids, all_log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        generator=generator
                )

            all_latents = torch.stack(all_latents, dim=1)  # (batch_size, window_size + 1, 16, 96, 96)
            all_log_probs = torch.stack(all_log_probs, dim=1)  # shape after stack (batch_size, window_size)

            timesteps = pipeline.scheduler.get_window_timesteps()  # (window_size, )
            noise_levels = torch.as_tensor([pipeline.scheduler.get_noise_level_for_timestep(t) for t in timesteps], device=accelerator.device)  # (window_size, )
            timesteps = timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, window_size)
            noise_levels = noise_levels.repeat(config.sample.batch_size, 1)  # (batch_size, window_size)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
            # yield to to make sure reward computation starts
            time.sleep(0)

            # Wait for reward computation directly
            rewards, rewards_metadata = rewards.result()

            rewards = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "noise_levels": noise_levels,
                    "latents": all_latents[:, :-1],  # each entry is the latent at timestep t - 1 (init latents for 0)
                    "next_latents": all_latents[:, 1:],  # each entry is the latent at timestep t
                    "log_probs": all_log_probs,
                    "rewards": rewards
                }
            )
            if config.enable_mem_log:
                memory_profiler.track_samples(samples, f"sampling")
                memory_profiler.snapshot(f"epoch_{epoch}_after_sampling_batch_{i}")

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }
        # for key, value in samples.items():
        #     if isinstance(value, torch.Tensor):
        #         print(key, 'has shape', value.shape)

        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value).cpu().numpy() for key, value in samples["rewards"].items()}
        # log rewards and images
        if accelerator.is_main_process:
            logging_platform.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items()},
                },
                step=global_step,
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = tokenizers[1].batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

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
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, *advantages.shape[1:])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            # for key, value in gathered_rewards.items():
            #     print(key, ": ", value)

            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape

        # A assertion to ensure the number of timesteps is consistent
        assert num_timesteps == num_train_timesteps
        if config.enable_mem_log:
            memory_profiler.snapshot(f"epoch_{epoch}_before_training")
        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size)
            samples = {k: v[perm] for k, v in samples.items()}

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size // config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            info = defaultdict(list)

            for i, sample in tqdm(
                list(enumerate(samples_batched)),
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

