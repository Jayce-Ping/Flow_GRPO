export CUDA_VISIBLE_DEVICES=0,2,3,4

export WANDB_API_KEY="66795f41320baafdbf8b4a19b62dce232ded0c2e"

# 4 GPU
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=4 --main_process_port 29501 scripts/train_flux.py --config config/grpo.py:consistency_flux_4gpu