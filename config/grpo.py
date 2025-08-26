import ml_collections
import os
import math
import base

# import imp
# base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

FLUX_MODEL_PATH = '/raid/data_qianh/jcy/hugging/models/FLUX.1-dev'
SD3_MODEL_PATH = "/raid/data_qianh/jcy/hugging/models/stable-diffusion-3.5-medium"

# --------------------------------------------------base------------------------------------------------------------
def compressibility():
    config = base.get_config()

    config.pretrained.model = SD3_MODEL_PATH
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.max_sequence_length = 512

    config.use_lora = True
    config.sample.use_sliding_window = False

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True


    # resume training
    config.resume_from_id = None
    config.resume_from_step = None
    config.resume_from_epoch = None
    config.project_name = None
    return config

# --------------------------------------------------------SD3------------------------------------------------------

def general_ocr_sd3():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = SD3_MODEL_PATH
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.use_history = False
    # Whether to use the same noise for the same prompt
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = SD3_MODEL_PATH
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.use_history = False
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = f'logs/geneval/sd3.5-M'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = SD3_MODEL_PATH
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.use_history = False
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def clipscore_sd3():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = SD3_MODEL_PATH
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.02
    config.sample.global_std = True
    config.sample.use_history = False
    config.sample.same_latent = True
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/clipscore/sd3.5-M'
    config.reward_fn = {
        "clipscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3_s1():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = SD3_MODEL_PATH
    config.sample.num_steps = 10
    config.sample.train_num_steps = 1
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.batch_size = 1
    config.sample.num_image_per_prompt = 24
    config.sample.mini_num_image_per_prompt = 6
    config.sample.num_batches_per_epoch = 4
    config.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-4
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.use_history = False
    config.sample.noise_level = 5
    config.train.ema = True
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/sd3.5-M-s1'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def general_ocr_sd3_4gpu():
    gpu_number = 4
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = SD3_MODEL_PATH
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.use_history = False
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3_4gpu():
    gpu_number=4
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = SD3_MODEL_PATH
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.use_history = False
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def general_ocr_sd3_1gpu():
    gpu_number = 1
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = SD3_MODEL_PATH
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.batch_size = 8
    config.sample.num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = int(8/(gpu_number*config.sample.batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.use_history = False
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def consistency_sd3_2gpu():
    gpu_number = 2
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/T2IS")

    # sd3.5 medium
    config.pretrained.model = SD3_MODEL_PATH
    config.sample.num_steps = 20
    config.sample.eval_num_steps = 20
    config.sample.guidance_scale = 4.5

    config.resolution = 1024
    config.max_sequence_length = 512
    config.sample.batch_size = 1
    config.sample.num_image_per_prompt = 24

    config.sample.unique_sample_num_per_epoch = 32 # Number of unique prompts used in each epoch
    # Number of unique samples per batch (gathing batches from all devices as one), a float number, maybe less than 1
    config.sample.unique_sample_num_per_batch = gpu_number * config.sample.batch_size / config.sample.num_image_per_prompt
    config.sample.num_batches_per_epoch = int(config.sample.unique_sample_num_per_epoch / config.sample.unique_sample_num_per_batch)

    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.test_batch_size = 8

    config.train.batch_size = config.sample.batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.use_history = False
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 10 # epoch
    config.eval_freq = 10
    config.save_dir = 'logs/consistency/sd3.5-M'
    config.reward_fn = {
        "consistency_score": 1.0,
    }
    
    config.prompt_fn = "geneval"
    config.resume_from_id = 'i9x38m4z'

    config.per_prompt_stat_tracking = True
    return config

def consistency_sd3_4gpu():
    gpu_number = 4
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/T2IS")

    # sd3.5 medium
    config.pretrained.model = SD3_MODEL_PATH
    config.sample.num_steps = 20
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 1024
    config.max_sequence_length = 512
    config.sample.batch_size = 2
    config.sample.num_image_per_prompt = 24

    config.sample.unique_sample_num_per_epoch = 48 # Number of unique prompts used in each epoch
    # Number of unique samples per batch (gathing batches from all devices as one), a float number, maybe less than 1
    config.sample.unique_sample_num_per_batch = gpu_number * config.sample.batch_size / config.sample.num_image_per_prompt
    config.sample.num_batches_per_epoch = int(config.sample.unique_sample_num_per_epoch / config.sample.unique_sample_num_per_batch)

    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.test_batch_size = 8

    config.train.batch_size = config.sample.batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.use_history = False
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/consistency/sd3.5-M'
    config.reward_fn = {
        "consistency_score": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

# -----------------------------------------------------------Flux---------------------------------------------------------------

def pickscore_flux_8gpu():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # flux
    config.pretrained.model = FLUX_MODEL_PATH
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5

    config.resolution = 512
    config.sample.batch_size = 3
    config.sample.num_image_per_prompt = 24
    
    config.sample.unique_sample_num_per_epoch = 48 # Number of unique prompts used in each epoch
    # Number of unique samples per batch (gathing batches from all devices as one), a float number, maybe less than 1
    config.sample.unique_sample_num_per_batch = gpu_number * config.sample.batch_size / config.sample.num_image_per_prompt
    config.sample.num_batches_per_epoch = int(config.sample.unique_sample_num_per_epoch / config.sample.unique_sample_num_per_batch)

    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

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

def consistency_flux_8gpu():
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/T2IS")

    # flux
    config.pretrained.model = FLUX_MODEL_PATH
    config.sample.num_steps = 20
    config.sample.eval_num_steps = 20
    config.sample.guidance_scale = 3.5

    # Sliding Window Scheduler
    config.sample.use_sliding_window = False
    config.sample.window_size = 4
    config.sample.left_boundary = 0

    config.resolution = 1024
    config.max_sequence_length = 512
    config.sample.batch_size = 1
    config.sample.num_image_per_prompt = 32
    config.sample.unique_sample_num_per_epoch = 48 # Number of unique prompts used in each epoch
    config.sample.sample_num_per_epoch = math.lcm(
        config.sample.num_image_per_prompt * config.sample.unique_sample_num_per_epoch,
        gpu_number * config.sample.batch_size
    ) # Total number of sample on all processes, to make sure all unique prompts are includede at least `num_image_per_prompt` times.

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

    config.test_batch_size = 8

    config.train.batch_size = config.sample.batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.train.ema = True
    config.sample.global_std = True
    config.sample.use_history = False
    config.sample.same_latent = False
    config.sample.noise_level = 0.9
    config.save_freq = 20 # epoch
    config.eval_freq = 20 # 0 for no eval applied
    config.save_dir = 'logs/consistency/flux-4gpu'
    config.reward_fn = {
        "consistency_score": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True

    # config.train.lora_path = 'logs/consistency/flux-4gpu/checkpoints/8-21-checkpoint-60/lora'
    config.project_name = 'FlowGRPO-Flux'

    return config

def consistency_flux_4gpu():
    gpu_number = 4
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/T2IS")

    # Sliding Window Scheduler
    config.sample.use_sliding_window = True
    config.sample.window_size = 4
    config.sample.left_boundary = 2

    # flux
    config.pretrained.model = FLUX_MODEL_PATH
    config.sample.num_steps = 20
    config.sample.eval_num_steps = 20
    config.sample.guidance_scale = 3.5

    config.resolution = 1024
    config.max_sequence_length = 512

    config.sample.batch_size = 1
    config.sample.num_image_per_prompt = 16
    config.sample.unique_sample_num_per_epoch = 32 # Number of unique prompts used in each epoch
    config.sample.sample_num_per_epoch = math.lcm(
        config.sample.num_image_per_prompt * config.sample.unique_sample_num_per_epoch,
        gpu_number * config.sample.batch_size
    ) # Total number of sample on all processes, to make sure all unique prompts are includede at least `num_image_per_prompt` times.

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

    config.test_batch_size = 8

    config.train.batch_size = config.sample.batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.train.ema = True
    config.sample.global_std = True
    config.sample.use_history = False
    config.sample.same_latent = False
    config.sample.noise_level = 0.9
    config.save_freq = 20 # epoch
    config.eval_freq = 20 # 0 for no eval applied
    config.save_dir = 'logs/consistency/flux-4gpu'
    config.reward_fn = {
        "consistency_score": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True

    # config.train.lora_path = 'logs/consistency/flux-4gpu/checkpoints/8-21-checkpoint-60/lora'
    config.project_name = 'FlowGRPO-Flux'

    return config


def consistency_flux_7gpu():
    gpu_number = 7 # Save one gpu to deploy vllm for consistency scoring
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/T2IS")

    # Sliding Window Scheduler
    config.sample.use_sliding_window = True
    config.sample.window_size = 4
    config.sample.left_boundary = 2

    # flux
    config.pretrained.model = FLUX_MODEL_PATH
    config.sample.num_steps = 20
    config.sample.eval_num_steps = 20
    config.sample.guidance_scale = 3.5

    config.resolution = 1024
    config.max_sequence_length = 512

    config.sample.batch_size = 1
    config.sample.num_image_per_prompt = 16
    config.sample.unique_sample_num_per_epoch = 35 # Number of unique prompts used in each epoch
    config.sample.sample_num_per_epoch = math.lcm(
        config.sample.num_image_per_prompt * config.sample.unique_sample_num_per_epoch,
        gpu_number * config.sample.batch_size
    ) # Total number of sample on all processes, to make sure all unique prompts are includede at least `num_image_per_prompt` times.

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

    config.test_batch_size = 8
    config.train.batch_size = config.sample.batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.use_history = False
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.save_freq = 15 # epoch
    config.eval_freq = 15 # -1 for no eval applied
    config.save_dir = 'logs/consistency/flux-7gpu'
    config.reward_fn = {
        "consistency_score": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True

    config.project_name = 'FlowGRPO-Flux'
    return config

def counting_flux_kontext():
    gpu_number=28
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting_edit")

    # sd3.5 medium
    config.pretrained.model = "black-forest-labs/FLUX.1-Kontext-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 2.5

    config.resolution = 512
    config.sample.batch_size = 3
    config.sample.num_image_per_prompt = 21
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.test_batch_size = 2 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

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
    config.save_dir = 'logs/counting_edit/flux_kontext'
    config.reward_fn = {
        "image_similarity": 0.5,
        "geneval": 0.5,
    }
    config.per_prompt_stat_tracking = True
    return config

def get_config(name):
    return globals()[name]()
