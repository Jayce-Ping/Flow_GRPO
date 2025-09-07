# export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY="66795f41320baafdbf8b4a19b62dce232ded0c2e"
# export WANDB_MODE=offline

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes=7 \
    --main_process_port 29501 \
    scripts/train_flux_flexible_size.py \
    --config config/grpo.py:grid_consistency_clip_flux