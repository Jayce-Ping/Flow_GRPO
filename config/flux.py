import ml_collections
import os
import math
from importlib.util import spec_from_file_location, module_from_spec
import inspect

import numpy as np
from scipy.stats import gmean, hmean
import torch
from datetime import datetime

time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

spec = spec_from_file_location('base', os.path.join(os.path.dirname(__file__), "base.py"))
base = module_from_spec(spec)
spec.loader.exec_module(base)

def get_gpu_count():
    """
        Get gpu number
    """
    # 1. Get CUDA_VISIBLE_DEVICES first
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_visible = os.environ['CUDA_VISIBLE_DEVICES']
        if cuda_visible:
            return len(cuda_visible.split(','))
    
    # 2. Use torch
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    
    return 1


FLUX_MODEL_PATH = "black-forest-labs/FLUX.1-dev"
# FLUX_MODEL_PATH = "/root/siton-data-51d3ce9aba3246f88f64ea65f79d5133/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
# SAVE_DIR = 'logs'
# SAVE_DIR = '/scratch/users/astar/ares/cp3jia/Flux_GRPO/logs'
# SAVE_DIR = '/root/siton-tmp/Flux_GRPO/logs'
# SAVE_DIR = '/home/hangwei/storage/jcy/Flux_GRPO/logs'
# SAVE_DIR = '/home/users/astar/cfar/stuchengyou/jcy/Flux_GRPO/logs'
SAVE_DIR = '/home/users/astar/cfar/stuchengyou/jcy/Paco-GRPO/logs'
# --------------------------------------------------base------------------------------------------------------------
def compressibility():
    config = base.get_config()

    config.pretrained.model = FLUX_MODEL_PATH
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.enable_mem_log = True
    config.use_lora = True

    # Sampling
    config.sample.noise_steps = [1]
    config.sample.merge_step = 0
    config.sample.use_sliding_window = False
    config.sample.left_boundary = 0
    config.sample.window_size = 20
    config.sample.batch_size = 1
    config.sample.reward_batch_size = config.sample.batch_size
    config.sample.num_steps = 20
    config.sample.num_images_per_prompt = 3
    config.sample.max_group_size = 16
    config.sample.num_batches_per_epoch = 4
    config.sample.guidance_scale = 3.5
    config.sample.cps = False
    config.sample.noise_level = 0.7
    config.sample.global_std = False
    config.sample.subfig_permutation = False
    config.sample.max_from_same_source = None

    # Training
    config.train.loss_type = 'ppo'
    config.enable_flexible_size = False
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 2
    config.train.gradient_step_per_epoch = 2
    config.enable_gradient_checkpointing = True
    config.train.nft_beta = 1
    config.train.decay_type = 1 if 'nft' in config.train.loss_type else 0
    config.train.timesteps = None
    config.train.guidance_scale = 3.5
    config.train.log_tame_delta = None

    # Testing
    config.test.save_eval_images = True
    config.test.batch_size = 4
    config.test.num_steps = 20
    config.test.merge_step = 1

    # prompting
    config.prompt_fn = "general_ocr"
    config.max_sequence_length = 512

    # rewards
    config.train.reward_fn = {"jpeg_compressibility": 1}
    config.train.aggregate_fn = None
    config.test.reward_fn = {"jpeg_compressibility": 1}
    config.test.aggregate_fn = None
    config.per_prompt_stat_tracking = True

    # resume training
    config.resume_from_id = None
    config.resume_from_step = None
    config.resume_from_epoch = None
    config.project_name = 'Paco-GRPO'
    return config

# --------------------------------------------------Some general aggregate functions------------------------------------------------------------
# Default: none (means simple weighted sum)

# Geometric mean aggregate function
def geometric_mean(**kwargs):
    values = [v for v in kwargs.values() if v is not None]
    gm = gmean(values)
    return gm

def harmonic_mean(**kwargs):
    values = [v for v in kwargs.values() if v is not None]
    hm = hmean(values)
    return hm

def get_log_tamed_aggregate_fn(delta: float = 0.2, epsilon: float = 1e-4):
    def agg_fn(**kwargs):
        # Filter out None values
        values = np.stack(
            [v for v in kwargs.values() if v is not None],
            axis=0
        ) # (num_rewards, sample_num) or (num_rewards, num_time_steps, sample_num)
        # Compute mean and std for each group
        means = np.mean(values, axis=-1)
        stds = np.std(values, axis=-1)

        h = stds / (means + epsilon)
        # Apply log transformation to values those rewards with high h values
        values[h > delta] = np.log(1 + values[h > delta])
        # Aggregate by summation
        aggregated = np.sum(values, axis=0)
        return aggregated

    return agg_fn

def standardize_aggregate_fn(**kwargs):
    values = np.stack(
        [v for v in kwargs.values() if v is not None],
        axis=0
    ) # (num_rewards, sample_num) or (num_rewards, num_time_steps, sample_num)
    means = np.mean(values, axis=-1, keepdims=True)
    stds = np.std(values, axis=-1, keepdims=True) + 1e-4
    standardized = (values - means) / stds
    aggregated = np.sum(standardized, axis=0)
    return aggregated


# -----------------------------------------------------------Flux---------------------------------------------------------------
def flux_log_consistency_clip():
    weights: tuple[float, float] = (0.2, 0.8)
    run_name = f'{weights[1]}s+log(1 + {weights[0]}cot), small, group_std'
    save_dir_suffix = f'{weights[1]}s-log-{weights[0]}cot_ppo_10sde_train1_groupstd_train-small'
    resolution: int = 512
    prompt_template_version: int = 0
    global_std: bool = False 
    loss_type: str = 'ppo'
    assert resolution in [256, 384, 512, 768, 1024], f"Unsupported resolution: {resolution}"
    gpu_number = get_gpu_count()
    config = compressibility()
    dataset_map = {
        256: "dataset/T2IS/half_2by2_micro_train",
        384: "dataset/T2IS/half_2by2_mini_train",
        512: "dataset/T2IS/half_2by2_small_train",
        768: "dataset/T2IS/half_2by2_medium_train",
        1024: "dataset/T2IS/half_2by2"
    }
    config.dataset = os.path.join(os.getcwd(), dataset_map[resolution])
    config.resolution = resolution
    config.train.resolution = resolution
    config.test.resolution = 1024 # Keep test resolution to 1024 for evaluation

    config.enable_flexible_size = False
    config.prompt_fn = "geneval"
    config.pretrained.model = FLUX_MODEL_PATH
    config.enable_mem_log = False
    # config.logging_platform = "swanlab"

    config.run_name = run_name
    # Add time stamp to save dir suffix to avoid overwriting
    save_dir_suffix = f"{save_dir_suffix}-{time_stamp}"
    config.save_dir = os.path.join(SAVE_DIR, f'consistencyReward-subclip', save_dir_suffix)
    config.save_freq = 10 # epoch
    config.eval_freq = 10 # 0 for no eval applied
    config.train.reward_fn = {
        "consistency_score": weights[0],
        "subfig_clipT" : weights[1]
    }
    config.train.reward_fn_kwargs = {
        'model': 'ConsistencyReward-7B-CoT-01',
        'port': 8000,
        "prompt_template_version": prompt_template_version
    }
    agg_fn = None # Use default weighted sum

    config.train.log_tame_delta = delta
    config.train.aggregate_fn_code = inspect.getsource(agg_fn) if agg_fn is not None else None
    config.train.aggregate_fn = agg_fn

    config.test.reward_fn = config.train.reward_fn
    config.test.reward_fn_kwargs = config.train.reward_fn_kwargs
    config.test.aggregate_fn_code = config.train.aggregate_fn_code
    config.test.aggregate_fn = config.train.aggregate_fn

    # Testing
    config.test.save_eval_images = True
    config.test.batch_size = 5
    config.test.num_steps = 20
    config.test.merge_step = 0

    # Sampling
    ## sliding window scheduler
    config.sample.global_std = global_std
    config.sample.use_history = False
    config.sample.same_latent = False
    config.sample.guidance_scale = 3.5
    config.sample.subfig_permutation = False

    config.sample.cps = False
    config.sample.num_steps = 10
    config.sample.noise_steps = [1]
    config.sample.noise_level = 0.7
    config.sample.merge_step = 0

    ## batches
    config.enable_gradient_checkpointing = False
    config.sample.batch_size = 1
    config.sample.reward_batch_size = min(config.sample.batch_size * 4, 8)
    config.sample.num_images_per_prompt = 16
    config.sample.max_group_size = config.sample.num_images_per_prompt or 16
    unique_sample_num_range = range(42, 50)
    # config.sample.unique_sample_num_per_epoch = 48 # Number of unique prompts used in each epoch all gathered
    config.sample.unique_sample_num_per_epoch = None # Number of unique prompts used in each epoch all gathered

    for num in unique_sample_num_range:
        total_samples = num * config.sample.num_images_per_prompt
        if total_samples % (gpu_number * config.sample.batch_size) == 0:
            config.sample.unique_sample_num_per_epoch = num
            break
    assert config.sample.unique_sample_num_per_epoch is not None, f"Cannot find proper unique_sample_num_per_epoch in range {list(unique_sample_num_range)}, please check your configuration!"
    
    config.sample.sample_num_per_epoch = math.lcm(
        config.sample.num_images_per_prompt * config.sample.unique_sample_num_per_epoch,
        gpu_number * config.sample.batch_size
    ) # Total number of samples on all processes

    # Update number of unique prompt per epoch
    config.sample.unique_sample_num_per_epoch = config.sample.sample_num_per_epoch // config.sample.max_group_size

    # number of batches per epoch per GPU
    config.sample.num_batches_per_epoch = int(config.sample.sample_num_per_epoch / (gpu_number * config.sample.batch_size))

    # Training
    config.train.param_noise_std = 0
    config.train.loss_type = loss_type
    config.train.batch_size = config.sample.batch_size
    config.train.learning_rate = 3e-4
    config.train.gradient_step_per_epoch = 1 if 'nft' in config.train.loss_type else 2
    assert config.sample.num_batches_per_epoch % config.train.gradient_step_per_epoch == 0, f"""Make sure num_batches_per_epoch is divisible by gradient_step_per_epoch."""
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // config.train.gradient_step_per_epoch
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 1
    config.train.guidance_scale = 3.5
    config.train.timesteps = config.sample.noise_steps # Train on all noise steps
    config.train.beta = 0
    config.train.nft_beta = 1
    config.train.decay_type = 1 if 'nft' in config.train.loss_type else 0
    config.train.ema = True
    config.per_prompt_stat_tracking = True

    return config

def flux_log_consistency_clip_param_noisy():
    weights: tuple[float, float] = (0.2, 0.8)
    run_name = f'{weights[1]}s+log(1 + {weights[0]}cot), small, group_std, param-noisy-01'
    save_dir_suffix = f'{weights[1]}s-log-{weights[0]}cot_ppo_10sde_train1_groupstd_train-small_param-noisy-01'
    resolution: int = 512
    prompt_template_version: int = 0
    global_std: bool = False 
    loss_type: str = 'ppo'
    param_noise_std: float = 0.01
    assert resolution in [256, 384, 512, 768, 1024], f"Unsupported resolution: {resolution}"
    gpu_number = get_gpu_count()
    config = compressibility()
    dataset_map = {
        256: "dataset/T2IS/half_2by2_micro_train",
        384: "dataset/T2IS/half_2by2_mini_train",
        512: "dataset/T2IS/half_2by2_small_train",
        768: "dataset/T2IS/half_2by2_medium_train",
        1024: "dataset/T2IS/half_2by2"
    }
    config.dataset = os.path.join(os.getcwd(), dataset_map[resolution])
    config.resolution = resolution
    config.train.resolution = resolution
    config.test.resolution = 1024 # Keep test resolution to 1024 for evaluation

    config.enable_flexible_size = False
    config.prompt_fn = "geneval"
    config.pretrained.model = FLUX_MODEL_PATH
    config.enable_mem_log = False
    # config.logging_platform = "swanlab"

    config.run_name = run_name
    # Add time stamp to save dir suffix to avoid overwriting
    save_dir_suffix = f"{save_dir_suffix}-{time_stamp}"
    config.save_dir = os.path.join(SAVE_DIR, f'consistencyReward-subclip', save_dir_suffix)
    config.save_freq = 10 # epoch
    config.eval_freq = 10 # 0 for no eval applied
    config.train.reward_fn = {
        "consistency_score": weights[0],
        "subfig_clipT" : weights[1]
    }
    config.train.reward_fn_kwargs = {
        'model': 'ConsistencyReward-7B-CoT-01',
        'port': 8000,
        "prompt_template_version": prompt_template_version
    }
    agg_fn = None # Use default weighted sum

    config.train.log_tame_delta = delta
    config.train.aggregate_fn_code = inspect.getsource(agg_fn) if agg_fn is not None else None
    config.train.aggregate_fn = agg_fn

    config.test.reward_fn = config.train.reward_fn
    config.test.reward_fn_kwargs = config.train.reward_fn_kwargs
    config.test.aggregate_fn_code = config.train.aggregate_fn_code
    config.test.aggregate_fn = config.train.aggregate_fn

    # Testing
    config.test.save_eval_images = True
    config.test.batch_size = 5
    config.test.num_steps = 20
    config.test.merge_step = 0

    # Sampling
    ## sliding window scheduler
    config.sample.global_std = global_std
    config.sample.use_history = False
    config.sample.same_latent = False
    config.sample.guidance_scale = 3.5
    config.sample.subfig_permutation = False

    config.sample.cps = False
    config.sample.num_steps = 10
    config.sample.noise_steps = [1]
    config.sample.noise_level = 0.7
    config.sample.merge_step = 0

    ## batches
    config.enable_gradient_checkpointing = False
    config.sample.batch_size = 1
    config.sample.reward_batch_size = min(config.sample.batch_size * 4, 8)
    config.sample.num_images_per_prompt = 16
    config.sample.max_group_size = config.sample.num_images_per_prompt or 16
    unique_sample_num_range = range(42, 50)
    # config.sample.unique_sample_num_per_epoch = 48 # Number of unique prompts used in each epoch all gathered
    config.sample.unique_sample_num_per_epoch = None # Number of unique prompts used in each epoch all gathered

    for num in unique_sample_num_range:
        total_samples = num * config.sample.num_images_per_prompt
        if total_samples % (gpu_number * config.sample.batch_size) == 0:
            config.sample.unique_sample_num_per_epoch = num
            break
    assert config.sample.unique_sample_num_per_epoch is not None, f"Cannot find proper unique_sample_num_per_epoch in range {list(unique_sample_num_range)}, please check your configuration!"
    
    config.sample.sample_num_per_epoch = math.lcm(
        config.sample.num_images_per_prompt * config.sample.unique_sample_num_per_epoch,
        gpu_number * config.sample.batch_size
    ) # Total number of samples on all processes

    # Update number of unique prompt per epoch
    config.sample.unique_sample_num_per_epoch = config.sample.sample_num_per_epoch // config.sample.max_group_size

    # number of batches per epoch per GPU
    config.sample.num_batches_per_epoch = int(config.sample.sample_num_per_epoch / (gpu_number * config.sample.batch_size))

    # Training
    config.train.param_noise_std = param_noise_std
    config.train.loss_type = loss_type
    config.train.batch_size = config.sample.batch_size
    config.train.learning_rate = 3e-4
    config.train.gradient_step_per_epoch = 1 if 'nft' in config.train.loss_type else 2
    assert config.sample.num_batches_per_epoch % config.train.gradient_step_per_epoch == 0, f"""Make sure num_batches_per_epoch is divisible by gradient_step_per_epoch."""
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // config.train.gradient_step_per_epoch
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 1
    config.train.guidance_scale = 3.5
    config.train.timesteps = config.sample.noise_steps # Train on all noise steps
    config.train.beta = 0
    config.train.nft_beta = 1
    config.train.decay_type = 1 if 'nft' in config.train.loss_type else 0
    config.train.ema = True
    config.per_prompt_stat_tracking = True

    return config

def generate_ConsistencyReward_clip_config_for_resolution_exp(
    run_name: str,
    save_dir_suffix: str,
    resolution: int,
    prompt_template_version: int = 0,
    weights: tuple[float, float] = (0.2, 0.8),
    delta: float = 0.2,
    global_std: bool = False,
    loss_type: str = 'ppo',
    param_noise_std: float = 0,
):
    assert resolution in [256, 384, 512, 768, 1024], f"Unsupported resolution: {resolution}"
    gpu_number = get_gpu_count()
    config = compressibility()
    dataset_map = {
        256: "dataset/T2IS/half_2by2_micro_train",
        384: "dataset/T2IS/half_2by2_mini_train",
        512: "dataset/T2IS/half_2by2_small_train",
        768: "dataset/T2IS/half_2by2_medium_train",
        1024: "dataset/T2IS/half_2by2"
    }
    config.dataset = os.path.join(os.getcwd(), dataset_map[resolution])
    config.resolution = resolution
    config.train.resolution = resolution
    config.test.resolution = 1024 # Keep test resolution to 1024 for evaluation

    config.enable_flexible_size = False
    config.prompt_fn = "geneval"
    config.pretrained.model = FLUX_MODEL_PATH
    config.enable_mem_log = False
    # config.logging_platform = "swanlab"

    config.run_name = run_name
    # Add time stamp to save dir suffix to avoid overwriting
    save_dir_suffix = f"{save_dir_suffix}-{time_stamp}"
    config.save_dir = os.path.join(SAVE_DIR, f'consistencyReward-subclip', save_dir_suffix)
    config.save_freq = 10 # epoch
    config.eval_freq = 10 # 0 for no eval applied
    config.train.reward_fn = {
        "consistency_score": weights[0],
        "subfig_clipT" : weights[1]
    }
    config.train.reward_fn_kwargs = {
        'model': 'ConsistencyReward-7B-CoT-01',
        'port': 8000,
        "prompt_template_version": prompt_template_version
    }
    agg_fn = None # Use default weighted sum

    config.train.log_tame_delta = delta
    config.train.aggregate_fn_code = inspect.getsource(agg_fn) if agg_fn is not None else None
    config.train.aggregate_fn = agg_fn

    config.test.reward_fn = config.train.reward_fn
    config.test.reward_fn_kwargs = config.train.reward_fn_kwargs
    config.test.aggregate_fn_code = config.train.aggregate_fn_code
    config.test.aggregate_fn = config.train.aggregate_fn

    # Testing
    config.test.save_eval_images = True
    config.test.batch_size = 5
    config.test.num_steps = 20
    config.test.merge_step = 0

    # Sampling
    ## sliding window scheduler
    config.sample.global_std = global_std
    config.sample.use_history = False
    config.sample.same_latent = False
    config.sample.guidance_scale = 3.5
    config.sample.subfig_permutation = False

    config.sample.cps = False
    config.sample.num_steps = 10
    config.sample.noise_steps = [1]
    config.sample.noise_level = 0.7
    config.sample.merge_step = 0

    ## batches
    config.enable_gradient_checkpointing = False
    config.sample.batch_size = 1
    config.sample.reward_batch_size = min(config.sample.batch_size * 4, 8)
    config.sample.num_images_per_prompt = 16
    config.sample.max_group_size = config.sample.num_images_per_prompt or 16
    unique_sample_num_range = range(42, 50)
    # config.sample.unique_sample_num_per_epoch = 48 # Number of unique prompts used in each epoch all gathered
    config.sample.unique_sample_num_per_epoch = None # Number of unique prompts used in each epoch all gathered

    for num in unique_sample_num_range:
        total_samples = num * config.sample.num_images_per_prompt
        if total_samples % (gpu_number * config.sample.batch_size) == 0:
            config.sample.unique_sample_num_per_epoch = num
            break
    assert config.sample.unique_sample_num_per_epoch is not None, f"Cannot find proper unique_sample_num_per_epoch in range {list(unique_sample_num_range)}, please check your configuration!"
    
    config.sample.sample_num_per_epoch = math.lcm(
        config.sample.num_images_per_prompt * config.sample.unique_sample_num_per_epoch,
        gpu_number * config.sample.batch_size
    ) # Total number of samples on all processes

    # Update number of unique prompt per epoch
    config.sample.unique_sample_num_per_epoch = config.sample.sample_num_per_epoch // config.sample.max_group_size

    # number of batches per epoch per GPU
    config.sample.num_batches_per_epoch = int(config.sample.sample_num_per_epoch / (gpu_number * config.sample.batch_size))

    # Training
    config.train.param_noise_std = param_noise_std
    config.train.loss_type = loss_type
    config.train.batch_size = config.sample.batch_size
    config.train.learning_rate = 3e-4
    config.train.gradient_step_per_epoch = 1 if 'nft' in config.train.loss_type else 2
    assert config.sample.num_batches_per_epoch % config.train.gradient_step_per_epoch == 0, f"""Make sure num_batches_per_epoch is divisible by gradient_step_per_epoch."""
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // config.train.gradient_step_per_epoch
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 1
    config.train.guidance_scale = 3.5
    config.train.timesteps = config.sample.noise_steps # Train on all noise steps
    config.train.beta = 0
    config.train.nft_beta = 1
    config.train.decay_type = 1 if 'nft' in config.train.loss_type else 0
    config.train.ema = True
    config.per_prompt_stat_tracking = True

    return config

def consistencyReward_clip_ori():
    prompt_template_version = 0
    weights = (0.1, 1.0)
    delta = 0.2
    run_name = f'Auto-tame ({delta}), {weights[1]}s+{weights[0]}cot, ori, group_std'
    save_dir_suffix = f'10s-1cot_ppo_10sde_train1_groupstd_train-ori'
    resolution = 1024
    config = generate_ConsistencyReward_clip_config_for_resolution_exp(
        run_name=run_name,
        save_dir_suffix=save_dir_suffix,
        resolution=resolution,
        prompt_template_version=prompt_template_version,
        weights=weights,
        delta=delta
    )
    return config

def consistencyReward_clip_medium():
    prompt_template_version = 0
    weights = (0.1, 1.0)
    delta = 0.2
    run_name = f'Auto-tame ({delta}), {weights[1]}s+{weights[0]}cot, medium, group_std'
    save_dir_suffix = f'10s-1cot_ppo_10sde_train1_groupstd_train-medium'
    resolution = 768
    config = generate_ConsistencyReward_clip_config_for_resolution_exp(
        run_name=run_name,
        save_dir_suffix=save_dir_suffix,
        resolution=resolution,
        prompt_template_version=prompt_template_version,
        weights=weights,
        delta=delta
    )
    return config

def consistencyReward_clip_small_guard():
    prompt_template_version = 0
    weights = (0.2, 1.0)
    delta = 0.2
    run_name = f'Guard-GRPO, Auto-tame ({delta}), {weights[1]}s+{weights[0]}cot, small, group_std'
    save_dir_suffix = f'{weights[1]}s-{weights[0]}cot_guard_10sde_train1_groupstd_train-small'
    resolution = 512
    config = generate_ConsistencyReward_clip_config_for_resolution_exp(
        run_name=run_name,
        save_dir_suffix=save_dir_suffix,
        resolution=resolution,
        prompt_template_version=prompt_template_version,
        weights=weights,
        delta=delta,
        loss_type='guard_grpo'
    )
    return config

def consistencyReward_clip_small_guard_2():
    prompt_template_version = 0
    weights = (0.2, 0.8)
    delta = 0.2
    run_name = f'Guard-GRPO, Auto-tame ({delta}), {weights[1]}s+{weights[0]}cot, small, group_std'
    save_dir_suffix = f'{weights[1]}s-{weights[0]}cot_guard_10sde_train1_groupstd_train-small'
    resolution = 512
    config = generate_ConsistencyReward_clip_config_for_resolution_exp(
        run_name=run_name,
        save_dir_suffix=save_dir_suffix,
        resolution=resolution,
        prompt_template_version=prompt_template_version,
        weights=weights,
        delta=delta,
        loss_type='guard_grpo'
    )
    return config

def consistencyReward_clip_small_guard_3():
    prompt_template_version = 0
    weights = (0.2, 0.8)
    delta = 0.15
    run_name = f'Guard-GRPO, Auto-tame ({delta}), {weights[1]}s+{weights[0]}cot, small, group_std'
    save_dir_suffix = f'{weights[1]}s-{weights[0]}cot_guard_10sde_train1_groupstd_train-small'
    resolution = 512
    config = generate_ConsistencyReward_clip_config_for_resolution_exp(
        run_name=run_name,
        save_dir_suffix=save_dir_suffix,
        resolution=resolution,
        prompt_template_version=prompt_template_version,
        weights=weights,
        delta=delta,
        loss_type='guard_grpo'
    )
    return config

def consistencyReward_clip_small():
    prompt_template_version = 0
    weights = (0.2, 0.8)
    delta = 0.2
    run_name = f'Auto-tame ({delta}), {weights[1]}s+{weights[0]}cot, small, group_std'
    save_dir_suffix = f'{weights[1]}s-{weights[0]}cot_ppo_10sde_train1_groupstd_train-small'
    resolution = 512
    config = generate_ConsistencyReward_clip_config_for_resolution_exp(
        run_name=run_name,
        save_dir_suffix=save_dir_suffix,
        resolution=resolution,
        prompt_template_version=prompt_template_version,
        weights=weights,
        delta=delta
    )
    return config

def consistencyReward_clip_small_37():
    prompt_template_version = 0
    weights = (0.3, 0.7)
    delta = 0.2
    run_name = f'Auto-tame ({delta}), {weights[1]}s+{weights[0]}cot, small, group_std'
    save_dir_suffix = f'{weights[1]}s-{weights[0]}cot_ppo_10sde_train1_groupstd_train-small'
    resolution = 512
    config = generate_ConsistencyReward_clip_config_for_resolution_exp(
        run_name=run_name,
        save_dir_suffix=save_dir_suffix,
        resolution=resolution,
        prompt_template_version=prompt_template_version,
        weights=weights,
        delta=delta
    )
    return config

def consistencyReward_clip_mini():
    prompt_template_version = 0
    weights = (0.1, 1.0)
    delta = 0.2
    run_name = f'Auto-tame ({delta}), {weights[1]}s+{weights[0]}cot, mini, group_std'
    save_dir_suffix = f'10s-1cot_ppo_10sde_train1_groupstd_train-mini'
    resolution = 384
    config = generate_ConsistencyReward_clip_config_for_resolution_exp(
        run_name=run_name,
        save_dir_suffix=save_dir_suffix,
        resolution=resolution,
        prompt_template_version=prompt_template_version,
        weights=weights,
        delta=delta
    )
    return config

def consistencyReward_clip_micro():
    prompt_template_version = 0
    weights = (0.1, 1.0)
    delta = 0.2
    run_name = f'Auto-tame ({delta}), {weights[1]}s+{weights[0]}cot, micro, group_std'
    save_dir_suffix = f'10s-1cot_ppo_10sde_train1_groupstd_train-micro'
    resolution = 256
    config = generate_ConsistencyReward_clip_config_for_resolution_exp(
        run_name=run_name,
        save_dir_suffix=save_dir_suffix,
        resolution=resolution,
        prompt_template_version=prompt_template_version,
        weights=weights,
        delta=delta
    )
    return config

def consistencyReward_clip_small_no_tame():
    prompt_template_version = 0
    weights = (0.1, 1.0)
    delta = 0.2
    run_name = f'No-tame, {weights[1]}s+{weights[0]}cot, small, group_std'
    save_dir_suffix = f'10s-1cot_ppo_10sde_train1_groupstd_train-small-notame'
    resolution = 512
    config = generate_ConsistencyReward_clip_config_for_resolution_exp(
        run_name=run_name,
        save_dir_suffix=save_dir_suffix,
        resolution=resolution,
        prompt_template_version=prompt_template_version,
        weights=weights,
        delta=delta
    )
    return config

def get_config(name):
    return globals()[name]()
