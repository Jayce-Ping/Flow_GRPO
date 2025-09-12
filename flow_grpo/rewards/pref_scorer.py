import os
import re
import json
from socket import timeout
from typing import List, Tuple, Union
from io import BytesIO
import base64
import logging
import asyncio
from itertools import combinations
import math
import time

import torch
import numpy as np
import openai
from openai import OpenAI, AsyncOpenAI
from PIL import Image
from flow_grpo.rewards.utils import pil_image_to_base64, divide_image, extract_grid_info
from flow_grpo.rewards.utils import get_yes_cond_prob_from_completion

# VLLM log filter
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


def pref_score():
    client = AsyncOpenAI(
        api_key='dummy-key',
        base_url='http://127.0.0.1:8000/v1'
    )

    scorer = PrefScorer(
        client=client,
        model='Qwen2.5-VL-7B-Instruct',
        criteria_path='dataset/T2IS/prompt_consistency_criterion.json',
        max_concurrent=60, # Adjust based on the system's capabilities (especially when using vllm as local model server)
    )

    def _fn(images, prompts, metadatas):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        
        scores = asyncio.run(scorer(images, prompts, metadatas))
        return scores, {}

    return _fn


def build_messages(image1 : Image.Image, image2 : Image.Image, prompt: str):

    messages = [
        {
            "role": "system",
            "content": (
                "You are an evaluator. Compare two images and decide whether the first image "
                "demonstrates better overall consistency than the second image. "
                "Consistency includes: "
                "- Theme Consistency (same topic or scenario across sub-images) "
                "- Style Consistency (colors, rendering style coherent) "
                "- Logical Consistency (actions or objects connect logically without contradictions). "
                "Answer strictly with 'Yes' or 'No'. Do not provide explanations."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Image description: {prompt}"},
                {"type": "image_url", "image_url": {"url": pil_image_to_base64(image1)}},
                {"type": "image_url", "image_url": {"url": pil_image_to_base64(image2)}},
                {"type": "text", "text": "Question: Does the first image have better consistency than the second image?"}
            ],
        },
    ]
    return messages


class PrefScorer:
    def __init__(
            self,
            client: AsyncOpenAI,
            model='Qwen2.5-VL-7B-Instruct',
            criteria_path='prompt_consistency_criterion.json',
            async_mode=True,
            max_concurrent=60,
            max_retries=10,
            timeout=60
        ):
        self.client = client
        self.model = model
        self.async_mode = async_mode
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout = timeout

        with open(criteria_path, 'r') as f:
            self.criteria_data = json.load(f)

    async def __call__(self, images : list[Image.Image], prompts : list[str], metadata: list[dict]) -> Union[list[float], np.ndarray]:
        assert len(images) == len(prompts) == len(metadata), "Length of images, prompts, and metadata must be the same."
        # Group images and metadata by their prompts
        prompt_to_images = {}
        prompt_to_metadata = {}
        prompt_to_pos = {}
        for i, (img, prompt, meta) in enumerate(zip(images, prompts, metadata)):
            if prompt not in prompt_to_images:
                prompt_to_images[prompt] = []
                prompt_to_metadata[prompt] = []
                prompt_to_pos[prompt] = []
            prompt_to_images[prompt].append(img)
            prompt_to_metadata[prompt].append(meta)
            prompt_to_pos[prompt].append(i)

        # Process each group of images with the same prompt concurrently
        all_scores = np.zeros(len(images), dtype=float)
        
        # Create tasks for each prompt group
        tasks = []
        for prompt, imgs in prompt_to_images.items():
            meta = prompt_to_metadata[prompt][0]
            task = self.compute_group_rewards(prompt, imgs, meta)
            tasks.append((prompt, task))
        
        # Execute all groups concurrently
        results = await asyncio.gather(*[task for _, task in tasks])
        
        # Assign results back to all_scores
        for (prompt, _), group_rewards in zip(tasks, results):
            all_scores[prompt_to_pos[prompt]] = group_rewards

        return all_scores
        

    async def compute_group_rewards(self, prompt : str, images : list[Image.Image], metadata: dict) -> Union[list[float], np.ndarray]:
        async def compare_image_pair(image1, image2):
            # Symmetric comparison for better reliability
            completion1 = await self.compare_image(image1, image2, prompt)
            completion2 = await self.compare_image(image2, image1, prompt)
            prob1 = get_yes_cond_prob_from_completion(completion1, canonicalize=True)
            prob2 = get_yes_cond_prob_from_completion(completion2, canonicalize=True)
            return int(prob1 > prob2), int(prob2 > prob1)

        # Process all image pairs concurrently
        comparison_array = np.zeros((len(images), len(images)), dtype=int)
        tasks = []
        pairs = []
        
        for i, j in combinations(range(len(images)), 2):
            task = compare_image_pair(images[i], images[j])
            tasks.append(task)
            pairs.append((i, j))
        
        # Execute all comparisons concurrently
        results = await asyncio.gather(*tasks)
        
        # Fill the comparison array
        for (i, j), (win1, win2) in zip(pairs, results):
            comparison_array[i, j] = win1
            comparison_array[j, i] = win2
        
        # Sum up wins for each image
        scores = comparison_array.sum(axis=1) / (max(1, len(images)-1))  # Normalize to [0, 1]
        return scores
    
    async def compare_image(
            self,
            image1 : Image.Image,
            image2 : Image.Image,
            prompt : str = "",
            top_logprobs: int = 20
        ) -> openai.ChatCompletion:
        global_semaphore = asyncio.Semaphore(self.max_concurrent)
        
        messages = build_messages(image1, image2, prompt)
        for attempt in range(self.max_retries):
            try:
                async with global_semaphore:
                    completion = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.0, # Deterministic result, no use for logprobs, actually.
                        max_completion_tokens=1,
                        logprobs=True,
                        top_logprobs=top_logprobs,
                        timeout=self.timeout
                    )
                break
            except Exception as e:
                print(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    completion = None

        return completion
    


def main():
    reward_fn = pref_score()