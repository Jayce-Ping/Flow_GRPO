import ml_collections
import os
import math
from importlib.util import spec_from_file_location, module_from_spec
import inspect

import numpy as np

spec = spec_from_file_location('base', os.path.join(os.path.dirname(__file__), "base.py"))
base = module_from_spec(spec)
spec.loader.exec_module(base)


FLUX_MODEL_PATH = "black-forest-labs/FLUX.1-dev"
# FLUX_MODEL_PATH = "/root/siton-data-51d3ce9aba3246f88f64ea65f79d5133/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
# SAVE_DIR = 'logs'
# SAVE_DIR = '/scratch/users/astar/ares/cp3jia/FlowGRPO/logs'
SAVE_DIR = '/root/siton-tmp/Flow_GRPO/logs'

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

    # Training
    config.enable_flexible_size = True
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2
    config.enable_gradient_checkpointing = True
    config.train.nft_beta = 1
    config.train.decay_type = 1

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
    config.project_name = 'ConsistencyNFT-Flux'
    return config

# -----------------------------------------------------------Flux---------------------------------------------------------------

def grid_consistency_clip_flux():
    gpu_number = 2
    config = compressibility()

    config.project_name = 'FlowGRPO-Flux'
    config.dataset = os.path.join(os.getcwd(), "dataset/T2IS/train_half_2by2")
    config.prompt_fn = "geneval"
    config.pretrained.model = FLUX_MODEL_PATH
    config.enable_mem_log = False
    config.logging_platform = "swanlab"

    config.enable_flexible_size = False
    config.resolution = 1024
    config.max_sequence_length = 512

    # Testing
    config.test.batch_size = 5
    config.test.num_steps = 20
    config.test.merge_step = 0

    # Sampling
    ## sliding window scheduler
    config.sample.cps = False
    config.sample.num_steps = 8
    config.sample.use_sliding_window = True
    config.sample.window_size = 1
    config.sample.left_boundary = 1
    config.sample.noise_steps = [1]
    config.sample.noise_level = 0.7
    config.sample.merge_step = 0
    config.sample.global_std = True
    config.sample.use_history = False
    config.sample.same_latent = False
    config.sample.guidance_scale = 3.5

    config.sample.cps = False
    config.sample.noise_level = 0.7
    config.sample.global_std = True

    ## batches
    config.enable_gradient_checkpointing = True
    config.sample.batch_size = 2
    config.sample.num_images_per_prompt = 3
    config.sample.unique_sample_num_per_epoch = 16 # Number of unique prompts used in each epoch
    config.sample.sample_num_per_epoch = math.lcm(
        config.sample.num_images_per_prompt * config.sample.unique_sample_num_per_epoch,
        gpu_number * config.sample.batch_size
    ) # Total number of sample on all processes, to make sure all unique prompts are included `num_images_per_prompt` times.

    # Update number of unique prompt per epoch and check balance
    unique_sample_num_per_epoch = config.sample.sample_num_per_epoch // config.sample.num_images_per_prompt
    num_images_per_prompt = config.sample.sample_num_per_epoch // config.sample.unique_sample_num_per_epoch
    assert unique_sample_num_per_epoch % gpu_number == 0, f"""Assure all samples of one prompt are on the same GPU."""
    config.sample.num_images_per_prompt = num_images_per_prompt
    config.sample.unique_sample_num_per_epoch = unique_sample_num_per_epoch

    config.sample.num_batches_per_epoch = int(config.sample.sample_num_per_epoch / (gpu_number * config.sample.batch_size))

    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."

    # Training
    config.train.batch_size = config.sample.batch_size
    config.train.learning_rate = 3e-4
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.train.nft_beta = 1
    config.train.decay_type = 1

    config.train.ema = True
    config.per_prompt_stat_tracking = True
    config.save_freq = 20 # epoch
    config.eval_freq = 20 # 0 for no eval applied

    config.reward_fn = {
        "grid_layout": 1.0,
        "consistency_score": 0.2,
        "subfig_clipT" : 0.8
    }
    def agg_fn(grid_layout : np.ndarray, consistency_score : np.ndarray, subfig_clipT : np.ndarray) -> np.ndarray:
        return grid_layout * consistency_score + subfig_clipT
    
    config.aggregate_fn_code = inspect.getsource(agg_fn) if agg_fn is not None else None

    config.aggregate_fn = agg_fn

    config.save_dir = os.path.join(SAVE_DIR, f'grid-consistency-subclip', f'flux-{gpu_number}gpu-2by2-half_grid-times-consistency-plus-clipT_8steps')

    return config

def get_config(name):
    return globals()[name]()
