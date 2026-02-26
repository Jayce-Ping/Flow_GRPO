accelerate launch \
    --config_file scripts/accelerate_configs/fsdp.yaml \
    --num_processes=8 \
    --main_process_port 29501 \
    scripts/train_bagel.py \
    --config config/grpo.py:pickscore_bagel_lora
