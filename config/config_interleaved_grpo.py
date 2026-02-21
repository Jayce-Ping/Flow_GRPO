"""
Config for Interleaved Multi-Turn Dialogue RL (Text-GRPO + Flow-GRPO).
"""
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # ── General ──
    config.run_name = "interleaved_grpo"
    config.logdir = "logs"
    config.save_dir = "checkpoints"
    config.seed = 42
    config.num_epochs = 200
    config.num_checkpoint_limit = 5
    config.mixed_precision = "bf16"
    config.allow_tf32 = True
    config.debug = False
    config.resume_from = ""
    config.save_freq = 10
    config.eval_freq = 5
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = False
    config.resolution = 512
    config.image_shapes = [512, 512]

    # ── Model ──
    config.pretrained = ml_collections.ConfigDict()
    config.pretrained.model = "ByteDance-Seed/BAGEL-7B-MoT"

    # ── LoRA ──
    config.use_lora = True
    config.lora_rank = 64
    config.lora_alpha = 128

    # ── Dataset ──
    config.dataset = "data/interleaved_dialogues"
    config.prompt_fn = "interleaved"

    # ── Reward ──
    config.reward_fn = "pickscore_score"
    config.per_prompt_stat_tracking = True

    # ── Sampling ──
    config.sample = ml_collections.ConfigDict()
    config.sample.train_batch_size = 4
    config.sample.test_batch_size = 2
    config.sample.num_batches_per_epoch = 1
    config.sample.num_image_per_prompt = 4
    config.sample.num_steps = 10           # Flow-matching denoising steps
    config.sample.noise_level = 0.7
    config.sample.guidance_scale = 4.0
    config.sample.eval_guidance_scale = 4.0
    config.sample.eval_num_steps = 24
    config.sample.same_latent = True
    config.sample.sde_window_size = 5
    config.sample.sde_window_range = [0, 10]
    config.sample.max_text_length = 256     # Max text tokens per turn
    config.sample.text_temperature = 0.7

    # ── Training ──
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 4
    config.train.num_inner_epochs = 1
    config.train.learning_rate = 1e-5
    config.train.adam_beta1 = 0.9
    config.train.adam_beta2 = 0.999
    config.train.adam_weight_decay = 0.01
    config.train.adam_epsilon = 1e-8
    config.train.max_grad_norm = 1.0

    # Flow-GRPO (image) hyperparams
    config.train.clip_range_lt = 0.2       # PPO clip lower
    config.train.clip_range_gt = 0.2       # PPO clip upper
    config.train.beta = 0.01               # KL penalty for flow-grpo
    config.train.adv_clip_max = 5.0

    # Text-GRPO hyperparams
    config.train.text_clip_range = 0.2
    config.train.text_beta = 0.01          # KL penalty for text GRPO

    # Loss weights
    config.train.text_loss_weight = 1.0
    config.train.image_loss_weight = 1.0

    # EMA
    config.train.ema = False

    return config