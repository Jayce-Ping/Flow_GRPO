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

# VLLM log filter
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


def pil_image_to_base64(image, format="JPEG"):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image/{format.lower()};base64,{encoded_image_text}"
    return base64_qwen

def divide_image(image, grid_info : tuple[int, int]):
    assert len(grid_info) == 2, "grid_info must be a tuple of two integers (a, b)"

    a, b = grid_info
    width, height = image.size

    grid_cells = []
    cell_width = width // a
    cell_height = height // b

    # 2x2 grid
    # | 1 | 2 |
    # | 3 | 4 |
    # [
    # (0, 0, cell_width, cell_height),
    # (cell_width, 0, 2 * cell_width, cell_height),
    # (0, cell_height, cell_width, 2 * cell_height),
    # (cell_width, cell_height, 2 * cell_width, 2 * cell_height)
    # ]

    # Row first, column second
    for i in range(b):
        for j in range(a):
            upper = i * cell_height
            left = j * cell_width
            right = left + cell_width
            lower = upper + cell_height
            grid_cells.append(image.crop((left, upper, right, lower)))

    return grid_cells

def extract_grid_info(prompt) -> tuple[int, int]:
    # Grid can be represented as int x int, or int ⨉ int. ⨉ has unicode \u2a09
    match = re.findall(r'(\d+)\s*[x⨉]\s*(\d+)', prompt)
    if len(match) == 0:
        return (1, 1)

    return (int(match[0][0]), int(match[0][1]))


def get_score_from_completion(completion : openai.ChatCompletion) -> float:
    if completion is None:
        return 0.0

    logprobs = completion.choices[0].logprobs
    if logprobs:
        # Use logprobs to compute, score = P('yes') / (P('yes') + P('no'))
        # score = 1 / (1 + exp(logprob('no') -  logprob('yes')))
        # Same formular for logits as well. Since the sum term will cancel out.
        # Use uppercase only here.
        token_logprobs = {t.token: t.logprob for t in logprobs.content[0].top_logprobs}
        yes_logprob = token_logprobs.get('Yes', float('-inf'))
        no_logprob = token_logprobs.get('No', float('-inf'))

        if yes_logprob == float('-inf') and no_logprob == float('-inf'):
            # When inf - inf encountered, give 0.0 score.
            score = 0.0 # 0.0
        else:
            diff = torch.tensor(yes_logprob - no_logprob, dtype=torch.float64)
            score = torch.sigmoid(diff).item()
    else:
        # log_prob cannot be derived here. How to calculate?
        # TODO
        score = 0.0

    return score


class ConsistencyScorer:
    def __init__(
            self,
            api_key='dummy_key',
            base_url='http://127.0.0.1:8000/v1',
            model='Qwen2.5-VL-7B-Instruct',
            criteria_path='prompt_consistency_criterion.json',
            async_mode=True,
            max_concurrent=12,  # 2x2 grid has 6 pair of images to compare. 12 for at most 2 batches at once.
            max_retries=10,
            timeout=60
        ):
        self.openai_api_key = api_key
        self.openai_base_url = base_url
        self.model = model
        self.async_mode = async_mode
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout = timeout

        if async_mode:
            self.client = AsyncOpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )
        else:
            self.client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )

        with open(criteria_path, 'r') as f:
            self.criteria_data = json.load(f)

    @torch.no_grad()
    async def __call__(self, images : list[Image.Image], prompts : list[str], metadatas : list[dict]) -> list[float]:
        assert len(prompts) == len(images), "Length of prompts and images must match"

        # Create a global semaphore for overall concurrency control
        global_semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_single_image(prompt, image, metadata):
            async with global_semaphore:
                criteria_info = self.criteria_data[metadata['idx']]
                dimensions = criteria_info.keys()
                dimension_scores = {k:0.0 for k in dimensions}
                
                # Compute scores for each prompt-image pair from different dimensions
                for dimension in dimensions:
                    # Get criteria for this dimension
                    dimension_criteria = criteria_info[dimension][0]
                    criteria_texts = [c_t for c_t in dimension_criteria.values() if c_t]

                    # [criteria1_scores : list[float], criteria2_scores : list[float], ...]
                    criterion_scores = []
                    for ct in criteria_texts:
                        scores = await self.compute_image_consistency(prompt, image, ct)
                        criterion_scores.append(scores)

                    # Compute the average score within each criterion
                    criterion_scores = [sum(scores) / len(scores) if scores else 0.0 for scores in criterion_scores]

                    # Compute the overall score for this dimension
                    overall_score = sum(criterion_scores) / len(criterion_scores) if criterion_scores else 0.0
                    dimension_scores[dimension] = overall_score

                # Compute average scores from each dimension
                return sum(dimension_scores.values()) / len(dimension_scores)

        # Process all images concurrently
        tasks = [
            process_single_image(prompt, image, metadata) 
            for prompt, image, metadata in zip(prompts, images, metadatas)
        ]
        
        final_scores = await asyncio.gather(*tasks)
        return final_scores
    
    async def compute_image_consistency(
            self,
            prompt : str,
            image : Image.Image,
            criteria_text : str,
            top_logprobs: int = 20
        ) -> list[float]:
        if self.async_mode:
            return await self._async_compute_image_consistency(prompt, image, criteria_text, top_logprobs)
        else:
            return self._sync_compute_image_consistency(prompt, image, criteria_text, top_logprobs)

    async def _async_compute_image_consistency(
            self,
            prompt : str,
            image : Image.Image,
            criteria_text : str,
            top_logprobs: int = 20
        ) -> list[float]:
        """
        Async version of compute_image_consistency with concurrency control.
        """
        async def process_image_pair(image1, image2):
            messages = [
                {
                    "role": "user",
                    "content":
                    [
                        {"type": "image_url", "image_url": {"url": pil_image_to_base64(image1)}},
                        {"type": "image_url", "image_url": {"url": pil_image_to_base64(image2)}},
                        {"type": "text", "text": f"Do images meet the following criteria? {criteria_text} Please answer Yes or No."},
                    ]
                }
            ]
            for attempt in range(self.max_retries):
                try:
                    completion = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.0, # Deterministic result, no use for logprobs, actually.
                        max_completion_tokens=1,
                        logprobs=True,
                        top_logprobs=top_logprobs,
                        timeout=self.timeout
                    )
                    return completion
                except Exception as e:
                    print(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        return None


        grid_info = extract_grid_info(prompt)
        sub_images = divide_image(image, grid_info)
        
        # Create tasks for all image pairs (no additional semaphore here since global control is in __call__)
        tasks = []
        for image1, image2 in combinations(sub_images, 2):
            task = process_image_pair(image1, image2)
            tasks.append(task)

        # Execute all tasks concurrently
        completions = await asyncio.gather(*tasks)

        return [get_score_from_completion(c) for c in completions]

    def _sync_compute_image_consistency(
            self,
            prompt : str,
            image : Image.Image,
            criteria_text : str,
            top_logprobs: int = 20
        ) -> list[float]:
        """
        Compute the consistency score of a image, for a given criterion.
        """
        completions = []
        grid_info = extract_grid_info(prompt)
        sub_images = divide_image(image, grid_info)
        for image1, image2 in combinations(sub_images, 2):
            messages = [
                {
                    "role": "user",
                    "content":
                    [
                        {"type": "image_url", "image_url": {"url": pil_image_to_base64(image1)}},
                        {"type": "image_url", "image_url": {"url": pil_image_to_base64(image2)}},
                        {"type": "text", "text": f"Do images meet the following criteria? {criteria_text} Please answer Yes or No."},
                    ]
                }
            ]

            for attempt in range(self.max_retries):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.0, # Deterministic result, no use for logprobs, actually.
                        max_completion_tokens=1,
                        logprobs=True,
                        top_logprobs=top_logprobs,
                        timeout=self.timeout
                    )
                    return completion
                except Exception as e:
                    print(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        return None

            completions.append(completion)

        return [get_score_from_completion(c) for c in completions]