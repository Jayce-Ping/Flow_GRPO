export WANDB_API_KEY="66795f41320baafdbf8b4a19b62dce232ded0c2e"

# 8 GPU
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux.py --config config/grpo.py:consistency_flux_8gpu