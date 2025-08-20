export CUDA_VISIBLE_DEVICES=4,5,6,7

export WANDB_API_KEY="66795f41320baafdbf8b4a19b62dce232ded0c2e"

accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:consistency_sd3_4gpu
