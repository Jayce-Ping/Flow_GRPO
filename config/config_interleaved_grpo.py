"""
Config for Interleaved Multi-Turn Dialogue RL (Text-GRPO + Flow-GRPO).
Follows flow_grpo/config/grpo.py structure with get_config(name) dispatch.
"""
import os
import ml_collections


# ─────────────────────── Base (mirrors compressibility()) ────────────────

def _base():
    """Base config shared by all variants, mirroring flow_grpo defaults."""
    config = ml_collections.ConfigDict()

    # ── General ──
    config.run_name = ""
    config.logdir = "logs"
    config.save_dir = "checkpoints"
    config.seed = 42
    config.num_epochs = 500
    config.num_checkpoint_limit = 5
    config.mixed_precision = "bf16"
    config.allow_tf32 = True
    config.debug = False
    config.resume_from = ""
    config.save_freq = 30
    config.eval_freq = 30
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    config.resolution = 512

    # ── Model ──
    config.pretrained = ml_collections.ConfigDict()
    config.pretrained.model = "ByteDance-Seed/BAGEL-7B-MoT"

    # ── LoRA ──
    config.use_lora = False
    config.lora_rank = 64
    config.lora_alpha = 128

    # ── Dataset ──
    config.dataset = os.path.join(os.getcwd(), "dataset/umm/")
    config.prompt_fn = "interleaved"

    # ── Reward (dict format, same as flow_grpo) ──
    config.reward_fn = {
        "pickscore": 1.0,
    }
    config.per_prompt_stat_tracking = True

    # ── Sampling ──
    config.sample = ml_collections.ConfigDict()
    config.sample.train_batch_size = 6
    config.sample.test_batch_size = 1
    config.sample.num_batches_per_epoch = 2
    config.sample.num_image_per_prompt = 16
    config.sample.num_steps = 15
    config.sample.eval_num_steps = 50
    config.sample.noise_level = 1.3
    config.sample.guidance_scale = 4.0
    config.sample.eval_guidance_scale = 4.0
    config.sample.same_latent = False
    config.sample.global_std = False
    config.sample.sde_window_size = 3
    config.sample.sde_window_range = (0, 7)    # (0, num_steps//2)

    # Text generation sampling
    config.sample.max_text_length = 256
    config.sample.text_temperature = 0.7
    config.sample.text_do_sample = True

    # ── Training ──
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 6
    config.train.gradient_accumulation_steps = 1
    config.train.num_inner_epochs = 1
    config.train.learning_rate = 1e-4
    config.train.adam_beta1 = 0.9
    config.train.adam_beta2 = 0.999
    config.train.adam_weight_decay = 0.01
    config.train.adam_epsilon = 1e-8
    config.train.max_grad_norm = 1.0

    # Flow-GRPO (image) — aligned with pickscore_bagel
    config.train.cfg = True
    config.train.clip_range_lt = 1e-5
    config.train.clip_range_gt = 1e-5
    config.train.beta = 0                # KL penalty for flow-grpo (0 = off)
    config.train.adv_clip_max = 5.0

    # Text-GRPO
    config.train.text_clip_range = 0.2
    config.train.text_beta = 0.01        # KL penalty for text GRPO

    # Loss weights
    config.train.text_loss_weight = 1.0
    config.train.image_loss_weight = 1.0

    # EMA
    config.train.ema = False

    return config


# ──────────────────── Full fine-tune (32 GPU) ────────────────────────────

def interleaved_grpo_full():
    """Interleaved GRPO, full fine-tune, 32 GPUs. Aligned with pickscore_bagel."""
    gpu_number = 32
    config = _base()

    config.run_name = "[interleaved-grpo-full]-32gpu"
    config.pretrained.model = "ByteDance-Seed/BAGEL-7B-MoT"
    config.use_lora = False

    # Sampling
    config.sample.num_steps = 15
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4.0
    config.sample.eval_guidance_scale = 4.0
    config.sample.noise_level = 1.3
    config.sample.same_latent = False
    config.sample.global_std = False

    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(
        48 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt)
    )  # =2 for 32 gpus
    config.sample.test_batch_size = 1

    config.sample.sde_window_size = 3
    config.sample.sde_window_range = (0, config.sample.num_steps // 2)

    # Training — match pickscore_bagel exactly
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = max(config.sample.num_batches_per_epoch // 2, 1)
    config.train.num_inner_epochs = 1
    config.train.clip_range_lt = 1e-5
    config.train.clip_range_gt = 1e-5
    config.train.beta = 0
    config.train.learning_rate = 1e-4
    config.mixed_precision = "bf16"

    # Text GRPO
    config.train.text_clip_range = 0.2
    config.train.text_beta = 0.01
    config.train.text_loss_weight = 1.0
    config.train.image_loss_weight = 1.0

    config.save_freq = 30
    config.eval_freq = 30
    config.save_dir = "logs/interleaved_grpo/full"

    config.reward_fn = {"pickscore": 1.0}
    config.per_prompt_stat_tracking = True
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True

    return config


# ──────────────────── LoRA fine-tune (8 GPU) ─────────────────────────────

def interleaved_grpo_lora():
    """Interleaved GRPO, LoRA, 8 GPUs. Aligned with pickscore_bagel_lora."""
    gpu_number = 8
    config = _base()

    config.run_name = "[interleaved-grpo-lora]-8gpu"
    config.pretrained.model = "ByteDance-Seed/BAGEL-7B-MoT"
    config.dataset = 'dataset/umm/'
    config.use_lora = True
    config.lora_rank = 64
    config.lora_alpha = 128

    # Sampling
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.cfg_text_scale = 1.0
    config.sample.cfg_img_scale = 1.0
    config.sample.noise_level = 1.3
    config.sample.same_latent = False
    config.sample.global_std = False

    config.resolution = 512
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(
        48 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt)
    )  # =4 for 8 gpus
    config.sample.test_batch_size = 1

    config.sample.sde_window_size = 1
    config.sample.sde_window_range = (0, config.sample.num_steps // 2)

    # Training — match pickscore_bagel_lora exactly
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = max(config.sample.num_batches_per_epoch // 2, 1)
    config.train.num_inner_epochs = 1
    config.train.clip_range_lt = 1e-5
    config.train.clip_range_gt = 1e-5
    config.train.beta = 0
    config.train.learning_rate = 1e-4
    config.mixed_precision = "bf16"

    # Text GRPO
    config.train.text_clip_range = 0.2
    config.train.text_beta = 0
    config.train.text_loss_weight = 1.0
    config.train.image_loss_weight = 1.0

    config.save_freq = 30
    config.eval_freq = 30
    config.save_dir = "logs/interleaved_grpo/lora"

    config.reward_fn = {"pickscore": 1.0}
    config.per_prompt_stat_tracking = True
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True

    return config


# ──────────────────── Config Dispatch ────────────────────────────────────

def get_config(name):
    """Dispatch config by name, matching flow_grpo/config/grpo.py pattern."""
    return globals()[name]()
