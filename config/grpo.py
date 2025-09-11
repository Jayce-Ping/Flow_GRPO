import ml_collections
import os
import math
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np

spec = spec_from_file_location('base', os.path.join(os.path.dirname(__file__), "base.py"))
base = module_from_spec(spec)
spec.loader.exec_module(base)


FLUX_MODEL_PATH = "black-forest-labs/FLUX.1-dev"
SD3_MODEL_PATH = "/raid/data_qianh/jcy/hugging/models/stable-diffusion-3.5-medium"

# --------------------------------------------------base------------------------------------------------------------
def compressibility():
    config = base.get_config()

    config.pretrained.model = SD3_MODEL_PATH
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.enable_mem_log = True
    config.use_lora = True

    # Sampling
    config.sample.use_sliding_window = False
    config.sample.left_boundary = 0
    config.sample.window_size = 20
    config.sample.batch_size = 1
    config.sample.num_steps = 20
    config.sample.num_image_per_prompt = 1
    config.sample.num_batches_per_epoch = 4
    config.sample.guidance_scale = 3.5

    # Training
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    config.test.batch_size = 4
    config.test.num_steps = 20

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
    config.project_name = None
    return config

# -----------------------------------------------------------Flux---------------------------------------------------------------

def test_flux():
    gpu_number = 2
    config = compressibility()
    config.logging_platform = "swanlab"

    config.dataset = os.path.join(os.getcwd(), "dataset/T2IS/train_all_2by2")

    # flux
    config.pretrained.model = FLUX_MODEL_PATH
    config.resolution = 512

    config.test.num_steps = 20
    config.test.batch_size = 4

    config.sample.use_sliding_window = True
    config.sample.window_size = 1
    config.sample.left_boundary = 1
    config.sample.num_steps = 3
    config.sample.guidance_scale = 3.5
    config.sample.batch_size = 1
    config.sample.num_image_per_prompt = 2
    config.sample.unique_sample_num_per_epoch = 4 # Number of unique prompts used in each epoch
    config.sample.sample_num_per_epoch = math.lcm(
        config.sample.num_image_per_prompt * config.sample.unique_sample_num_per_epoch,
        gpu_number * config.sample.batch_size
    ) # Total number of sample on all processes, to make sure all unique prompts are included `num_image_per_prompt` times.

    # Update number of unique prompt per epoch and check balance
    unique_sample_num_per_epoch = config.sample.sample_num_per_epoch // config.sample.num_image_per_prompt
    num_image_per_prompt = config.sample.sample_num_per_epoch // config.sample.unique_sample_num_per_epoch
    assert unique_sample_num_per_epoch == config.sample.unique_sample_num_per_epoch and num_image_per_prompt == config.sample.num_image_per_prompt, \
        f""" Current setting:
            config.sample.unique_sample_num_per_epoch={config.sample.unique_sample_num_per_epoch}
            config.sample.num_image_per_prompt={config.sample.num_image_per_prompt}
            requires total sample number per epoch to be multiplies of {config.sample.unique_sample_num_per_epoch}*{config.sample.num_image_per_prompt}={config.sample.unique_sample_num_per_epoch*config.sample.num_image_per_prompt},
            which is not a multiple of sample_batch_size*gpu_number={config.sample.batch_size*gpu_number} and will cause unbalanced sampling.
            Consider to set config.sample.unique_sample_num_per_epoch to be {unique_sample_num_per_epoch},
            or config.sample.num_image_per_prompt to be {num_image_per_prompt}.
        """
    config.sample.num_batches_per_epoch = int(config.sample.sample_num_per_epoch / (gpu_number * config.sample.batch_size))

    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."

    config.train.batch_size = config.sample.batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.use_history = False
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.save_freq = 0 # epoch
    config.eval_freq = 10
    config.save_dir = 'logs/test_run'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def subfig_clip_flux_2gpu():
    gpu_number = 2
    config = compressibility()
    # config.logging_platform = "swanlab"
    
    config.dataset = os.path.join(os.getcwd(), "dataset/T2IS/train_half_leq_4")

    config.sample.use_sliding_window = True
    config.sample.window_size = 2
    config.sample.left_boundary = 1

    # flux
    config.pretrained.model = FLUX_MODEL_PATH
    config.sample.num_steps = 10
    config.sample.guidance_scale = 3.5

    config.resolution = 1024

    config.sample.batch_size = 1
    config.sample.num_image_per_prompt = 16
    config.sample.unique_sample_num_per_epoch = 24 # Number of unique prompts used in each epoch
    config.sample.sample_num_per_epoch = math.lcm(
        config.sample.num_image_per_prompt * config.sample.unique_sample_num_per_epoch,
        gpu_number * config.sample.batch_size
    ) # Total number of sample on all processes, to make sure all unique prompts are included `num_image_per_prompt` times.

    # Update number of unique prompt per epoch and check balance
    unique_sample_num_per_epoch = config.sample.sample_num_per_epoch // config.sample.num_image_per_prompt
    num_image_per_prompt = config.sample.sample_num_per_epoch // config.sample.unique_sample_num_per_epoch
    assert unique_sample_num_per_epoch == config.sample.unique_sample_num_per_epoch and num_image_per_prompt == config.sample.num_image_per_prompt, \
        f""" Current setting:
            config.sample.unique_sample_num_per_epoch={config.sample.unique_sample_num_per_epoch}
            config.sample.num_image_per_prompt={config.sample.num_image_per_prompt}
            requires total sample number per epoch to be multiplies of {config.sample.unique_sample_num_per_epoch}*{config.sample.num_image_per_prompt}={config.sample.unique_sample_num_per_epoch*config.sample.num_image_per_prompt},
            which is not a multiple of sample_batch_size*gpu_number={config.sample.batch_size*gpu_number} and will cause unbalanced sampling.
            Consider to set config.sample.unique_sample_num_per_epoch to be {unique_sample_num_per_epoch},
            or config.sample.num_image_per_prompt to be {num_image_per_prompt}.
        """
    config.sample.num_batches_per_epoch = int(config.sample.sample_num_per_epoch / (gpu_number * config.sample.batch_size))

    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."

    config.test.batch_size = 6
    config.test.num_steps = 20

    config.train.batch_size = config.sample.batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.use_history = False
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.save_freq = 0 # epoch
    config.eval_freq = 0
    config.save_dir = 'logs/subfig_clipT/flux_2gpu'
    config.reward_fn = {
        "subfig_clipT": 1.0,
    }

    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_flux_8gpu():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # flux
    config.test.num_steps = 28
    config.test.batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.pretrained.model = FLUX_MODEL_PATH
    config.sample.num_steps = 6
    config.sample.guidance_scale = 3.5

    config.resolution = 512
    config.sample.batch_size = 3
    config.sample.num_image_per_prompt = 24
    
    config.sample.unique_sample_num_per_epoch = 48 # Number of unique prompts used in each epoch
    # Number of unique samples per batch (gathing batches from all devices as one), a float number, maybe less than 1
    config.sample.unique_sample_num_per_batch = gpu_number * config.sample.batch_size / config.sample.num_image_per_prompt
    config.sample.num_batches_per_epoch = int(config.sample.unique_sample_num_per_epoch / config.sample.unique_sample_num_per_batch)

    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."

    config.train.batch_size = config.sample.batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.use_history = False
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/flux-group24-8gpu'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def grid_consistency_clip_flux():
    gpu_number = 7
    config = compressibility()

    config.project_name = 'FlowGRPO-Flux'
    config.dataset = os.path.join(os.getcwd(), "dataset/T2IS/train_half_leq_4")
    config.prompt_fn = "geneval"
    config.pretrained.model = FLUX_MODEL_PATH

    config.resolution = 1024
    config.max_sequence_length = 512

    # Testing
    config.test.batch_size = 4
    config.test.num_steps = 20

    # Sampling
    ## sliding window scheduler
    config.sample.num_steps = 20
    config.sample.use_sliding_window = True
    config.sample.window_size = 2
    config.sample.left_boundary = 1
    config.sample.guidance_scale = 3.5

    ## batches
    config.sample.batch_size = 1
    config.sample.num_image_per_prompt = 16
    config.sample.unique_sample_num_per_epoch = 42 # Number of unique prompts used in each epoch
    config.sample.sample_num_per_epoch = math.lcm(
        config.sample.num_image_per_prompt * config.sample.unique_sample_num_per_epoch,
        gpu_number * config.sample.batch_size
    ) # Total number of sample on all processes, to make sure all unique prompts are included `num_image_per_prompt` times.

    # Update number of unique prompt per epoch and check balance
    unique_sample_num_per_epoch = config.sample.sample_num_per_epoch // config.sample.num_image_per_prompt
    num_image_per_prompt = config.sample.sample_num_per_epoch // config.sample.unique_sample_num_per_epoch
    assert unique_sample_num_per_epoch == config.sample.unique_sample_num_per_epoch and num_image_per_prompt == config.sample.num_image_per_prompt, \
        f""" Current setting:
            config.sample.unique_sample_num_per_epoch={config.sample.unique_sample_num_per_epoch}
            config.sample.num_image_per_prompt={config.sample.num_image_per_prompt}
            requires total sample number per epoch to be multiplies of {config.sample.unique_sample_num_per_epoch}*{config.sample.num_image_per_prompt}={config.sample.unique_sample_num_per_epoch*config.sample.num_image_per_prompt},
            which is not a multiple of sample_batch_size*gpu_number={config.sample.batch_size*gpu_number} and will cause unbalanced sampling.
            Consider to set config.sample.unique_sample_num_per_epoch to be {unique_sample_num_per_epoch},
            or config.sample.num_image_per_prompt to be {num_image_per_prompt}.
        """

    config.sample.num_batches_per_epoch = int(config.sample.sample_num_per_epoch / (gpu_number * config.sample.batch_size))

    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."

    # Training
    config.train.batch_size = config.sample.batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.train.ema = True
    config.sample.global_std = True
    config.per_prompt_stat_tracking = True
    config.sample.use_history = False
    config.sample.same_latent = False
    config.sample.noise_level = 0.9
    config.save_freq = 10 # epoch
    config.eval_freq = 10 # 0 for no eval applied

    config.reward_fn = {
        "grid_layout": 1.0,
        "consistency_score": 0.3,
        "subfig_clipT" : 0.7
    }
    def agg_fn(grid_layout, consistency_score, subfig_clipT):
        return grid_layout * (consistency_score + subfig_clipT)
    
    config.aggregate_fn = agg_fn

    # config.save_dir = 'logs/grid-consistency-subclip/flux-7gpu-train-half-leq-4'
    # config.save_dir = '/scratch/users/astar/ares/cp3jia/checkpoints/flow-grpo/grid-consistency-subclip/flux-7gpu-train-leq-4'
    config.save_dir = '/root/autodl-tmp/checkpoints/flowgrpo/grid-consistency-subclip/flux-7gpu-train-half-leq-4'

    return config

def get_config(name):
    return globals()[name]()
