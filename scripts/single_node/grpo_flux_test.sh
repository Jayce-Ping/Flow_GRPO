# export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY="66795f41320baafdbf8b4a19b62dce232ded0c2e"
export WANDB_MODE=offline

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes=1 \
    --main_process_port 29501 \
    scripts/train_flux.py \
    --config config/grpo.py:test_flux_1gpu

# accelerate launch \
#     --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
#     --num_processes=4 \
#     --main_process_port 29501 \
#     scripts/train_flux.py \
#     --config config/grpo.py:test_flux_4gpu
