export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

export WANDB_API_KEY="66795f41320baafdbf8b4a19b62dce232ded0c2e"

# 7 GPU
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=7 --main_process_port 29501 scripts/train_flux.py --config config/grpo.py:consistency_flux_7gpu