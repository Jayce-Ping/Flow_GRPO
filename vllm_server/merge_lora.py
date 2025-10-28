import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import os
import argparse


def main(
    lora_path: str,
    output_path: str
):
    # 指定基础模型路径（假设已下载或本地路径）
    base_model_path = "Qwen/Qwen2.5-VL-7B-Instruct"  # 或本地路径，如 "/path/to/Qwen2.5-VL-7B-Instruct"

    # 指定LoRA适配器路径
    # lora_path = "~/scratch/ConsistencyReward/qwen2_5vl-7b_mix/lora/sft/checkpoint-388"  # 替换为实际LoRA路径

    # 加载基础模型和tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        device_map="balanced"  # 自动分配设备
    )
    processor = AutoProcessor.from_pretrained(base_model_path)

    # 加载LoRA适配器
    model = PeftModel.from_pretrained(model, lora_path)

    # 合并LoRA权重到基础模型并卸载LoRA
    merged_model = model.merge_and_unload()

    # 保存合并后的模型
    # output_path = "~/scratch/models/ConsistencyReward-7B-Mix-epoch1"  # 输出路径
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    print("LoRA合并完成，模型已保存到:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA adapter")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged model")
    args = parser.parse_args()

    main(
        lora_path=args.lora_path,
        output_path=args.output_path
    )