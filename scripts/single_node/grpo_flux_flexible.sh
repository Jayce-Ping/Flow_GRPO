# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export WANDB_API_KEY="66795f41320baafdbf8b4a19b62dce232ded0c2e"
# export WANDB_MODE=disabled

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # Audo-set number of GPUs if not set
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    # Count number of GPUs from CUDA_VISIBLE_DEVICES
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes=$NUM_GPUS \
    --main_process_port 29501 \
    scripts/train_flux_flexible.py \
    --config config/grpo.py:grid_consistency_clip_flux