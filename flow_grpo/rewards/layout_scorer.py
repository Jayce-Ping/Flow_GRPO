import os
import re
import json
from typing import List, Tuple, Union
from io import BytesIO
import base64
import logging
import asyncio
import time
from unittest import result
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


class GridLayoutScorer:
    def __init__(
            self,
            client : AsyncOpenAI,
            model='Qwen2.5-VL-7B-Instruct',
            max_concurrent=12,
            max_retries=10,
            timeout=60
        ):
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout = timeout

        self.client = client

    @torch.no_grad()
    async def __call__(self, images : list[Image.Image], prompts : list[str], metadatas : list[dict]) -> list[float]:
        assert len(prompts) == len(images), "Length of prompts and images must match"

        # Create a global semaphore for overall concurrency control
        global_semaphore = asyncio.Semaphore(self.max_concurrent)

        # Process all images concurrently
        async def process_single_image(prompt, image, metadata):
            async with global_semaphore:
                return await self.compute_layout_score(prompt, image, metadata)

        # Process all images concurrently
        tasks = [
            process_single_image(prompt, image, metadata) 
            for prompt, image, metadata in zip(prompts, images, metadatas)
        ]

        final_scores = await asyncio.gather(*tasks)
        return final_scores
    
    async def compute_layout_score(
            self,
            prompt : str,
            image : Image.Image,
            metadata : dict,
            top_logprobs: int = 20,
            threshold = 0.9,
        ) -> float:
        grid_info = extract_grid_info(prompt)
        messages = [
            {
                "role": "user",
                "content":
                [
                    {"type": "image_url", "image_url": {"url": pil_image_to_base64(image)}},
                    {"type": "text", "text": f"Is it a {grid_info} grid layout image? Please answer Yes or No."},
                ]
            }
        ]

        for attempt in range(self.max_retries):
            try:
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=1e-6, # Low temperature may cause issue here.
                    max_completion_tokens=1,
                    logprobs=True,
                    top_logprobs=top_logprobs,
                    timeout=self.timeout
                )
            except Exception as e:
                print(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    completion = None

        if completion is None:
            return 0.0
        else:
            content = completion.choices[0].message.content.strip().lower()
            if 'yes' in content:
                return 1.0
            else:
                return 0.0
            yes_prob = get_yes_cond_prob_from_completion(completion, canonicalize=True)
            if yes_prob > threshold:
                return 1.0
            else:
                return 0.0
