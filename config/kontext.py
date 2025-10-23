import ml_collections
import os
import math
from importlib.util import spec_from_file_location, module_from_spec
import inspect

import numpy as np
from scipy.stats import gmean, hmean

spec = spec_from_file_location('base', os.path.join(os.path.dirname(__file__), "base.py"))
base = module_from_spec(spec)
spec.loader.exec_module(base)


FLUX_MODEL_PATH = "black-forest-labs/FLUX.1-Kontext-dev"
# SAVE_DIR = 'logs'
# SAVE_DIR = '/scratch/users/astar/ares/cp3jia/Flow_NFT/logs'
SAVE_DIR = '/root/siton-tmp/Flow_Kontext/logs'

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
    config.sample.num_steps = 20
    config.sample.num_images_per_prompt = 3
    config.sample.max_group_size = 16
    config.sample.num_batches_per_epoch = 4
    config.sample.guidance_scale = 3.5
    config.sample.cps = False
    config.sample.noise_level = 0.7
    config.sample.global_std = True
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

    # Testing
    config.test.batch_size = 4
    config.test.num_steps = 20
    config.test.merge_step = 1

    # prompting
    config.prompt_fn = "general_ocr"
    config.max_sequence_length = 512

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.aggregate_fn = None
    config.per_prompt_stat_tracking = True

    # resume training
    config.resume_from_id = None
    config.resume_from_step = None
    config.resume_from_epoch = None
    config.project_name = 'FluxKontext-GRPO'
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

# -----------------------------------------------------------Flux---------------------------------------------------------------

def editscore():
    gpu_number = 2
    config = compressibility()

    # config.dataset = os.path.join(os.getcwd(), "dataset/counting_edit")
    config.dataset = "/root/siton-tmp/EditScore-RL-Data"
    config.resolution = 512
    config.enable_flexible_size = False
    config.pretrained.model = FLUX_MODEL_PATH
    config.enable_mem_log = False
    config.logging_platform = "swanlab"

    config.run_name = 'Flux Kontext'
    config.save_dir = os.path.join(SAVE_DIR, f'editscore', 'flux_kontext')
    config.save_freq = 0 # epoch
    config.eval_freq = 0 # 0 for no eval applied
    config.reward_fn = {
        "edit_score": 1.0,
    }
    agg_fn = None
    config.aggregate_fn_code = inspect.getsource(agg_fn) if agg_fn is not None else None
    config.aggregate_fn = agg_fn

    # Testing
    config.test.batch_size = 5
    config.test.num_steps = 20
    config.test.merge_step = 0

    # Sampling
    ## sliding window scheduler
    config.sample.global_std = False
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
    config.sample.num_images_per_prompt = 16
    config.sample.unique_sample_num_per_epoch = 2 # Number of unique prompts used in each epoch all gathered

    config.sample.sample_num_per_epoch = math.lcm(
        config.sample.num_images_per_prompt * config.sample.unique_sample_num_per_epoch,
        gpu_number * config.sample.batch_size
    ) # Total number of samples on all processes

    # Update number of unique prompt per epoch and check balance
    unique_sample_num_per_epoch = config.sample.sample_num_per_epoch // config.sample.num_images_per_prompt
    assert unique_sample_num_per_epoch % gpu_number == 0, f"""Assure all samples of one prompt are on the same GPU."""
    config.sample.unique_sample_num_per_epoch = unique_sample_num_per_epoch

    # number of batches per epoch per GPU
    config.sample.num_batches_per_epoch = int(config.sample.sample_num_per_epoch / (gpu_number * config.sample.batch_size))

    # Training
    config.train.param_noise_std = 0
    config.train.loss_type = 'ppo'
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

def consistencyreward_for_editing():
    gpu_number = 2
    config = compressibility()

    # config.dataset = os.path.join(os.getcwd(), "dataset/counting_edit")
    config.dataset = "/root/siton-tmp/EditScore-RL-Data"
    config.resolution = 512
    config.enable_flexible_size = False
    config.pretrained.model = FLUX_MODEL_PATH
    config.enable_mem_log = False
    config.logging_platform = "swanlab"

    config.run_name = 'Flux Kontext'
    config.save_dir = os.path.join(SAVE_DIR, f'editscore', 'flux_kontext')
    config.save_freq = 10 # epoch
    config.eval_freq = 10 # 0 for no eval applied
    config.reward_fn = {
        "consistencyreward_for_editing": 1.0,
    }
    agg_fn = None
    config.aggregate_fn_code = inspect.getsource(agg_fn) if agg_fn is not None else None
    config.aggregate_fn = agg_fn

    # Testing
    config.test.batch_size = 5
    config.test.num_steps = 20
    config.test.merge_step = 0

    # Sampling
    ## sliding window scheduler
    config.sample.global_std = False
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
    config.sample.num_images_per_prompt = 16
    config.sample.unique_sample_num_per_epoch = 16 # Number of unique prompts used in each epoch all gathered

    config.sample.sample_num_per_epoch = math.lcm(
        config.sample.num_images_per_prompt * config.sample.unique_sample_num_per_epoch,
        gpu_number * config.sample.batch_size
    ) # Total number of samples on all processes

    # Update number of unique prompt per epoch and check balance
    unique_sample_num_per_epoch = config.sample.sample_num_per_epoch // config.sample.num_images_per_prompt
    assert unique_sample_num_per_epoch % gpu_number == 0, f"""Assure all samples of one prompt are on the same GPU."""
    config.sample.unique_sample_num_per_epoch = unique_sample_num_per_epoch

    # number of batches per epoch per GPU
    config.sample.num_batches_per_epoch = int(config.sample.sample_num_per_epoch / (gpu_number * config.sample.batch_size))

    # Training
    config.train.param_noise_std = 0
    config.train.loss_type = 'ppo'
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


def get_config(name):
    return globals()[name]()
