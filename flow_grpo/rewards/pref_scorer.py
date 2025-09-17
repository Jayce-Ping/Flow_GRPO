import os
import re
import json
from socket import timeout
from typing import List, Tuple, Union, Optional, Callable, Any, Awaitable, Literal
from io import BytesIO
import base64
import logging
import asyncio
from itertools import combinations, permutations
import math
import time

import torch
import numpy as np
import openai
from openai import OpenAI, AsyncOpenAI
from PIL import Image
from flow_grpo.utils import pil_image_to_base64, divide_image, extract_grid_info
from flow_grpo.utils import get_yes_cond_prob_from_completion

# VLLM log filter
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


def pref_score():
    client = AsyncOpenAI(
        api_key='dummy-key',
        base_url='http://127.0.0.1:8000/v1'
    )

    def _fn(images : List[Image.Image], prompts : List[str], metadatas : List[dict]) -> Tuple[np.ndarray, dict]:
        # Create the PrefScorer instance inside the function to avoid semaphore issues
        scorer = PrefScorer(
            client=client,
            model='Qwen2.5-VL-7B-Instruct',
            max_concurrent=100, # Adjust based on the system's capabilities (especially when using vllm as local model server)
        )
        scores = scorer(images, prompts, metadatas)
        return scores, {}

    return _fn

# ---------------------------------------Group Pairwise Comparison---------------------------------------
class GroupPairwiseComparator:
    """
        This class aims to provide a framework for pairwise comparison-based sorting algorithms.
        Different algorithms can be implemented as different methods.
    """
    def __init__(self, items : list, comparison_matrix=None, compare_fn : Optional[Callable[[int, int], Awaitable[int]]]=None):
        """
            Args:
                items (`List[Any]`) : group elements
                comparison_matrix (`Optional[np.ndarray]`) : comparison matrix `m`, where each `m[i,j]` is in `[0,1]`. `m[i,j] == 0` means `i` is 'smaller' than `j`.
                
        """
        self.n = len(items)
        self.items = items
        self.comparison_matrix = np.zeros((self.n, self.n), dtype=int) if comparison_matrix is None else np.array(comparison_matrix)
        self.compare_fn = compare_fn
        self._comparison_count = 0

    def reset_comparisons(self):
        """
            Reset the comparison matrix to all zeros.
        """
        self.comparison_matrix = np.zeros((self.n, self.n), dtype=int)
        self._comparison_count = 0

    def is_compared(self, i : int, j : int) -> bool:
        """
            Check if items[i] and items[j] have been compared.
        """
        assert i != j, ValueError(f"Cannot check comparison for the same item {i} == {j}.")
        return self.comparison_matrix[i][j] != 0 or self.comparison_matrix[j][i] != 0

    def add_comparison(self, i : int, j : int, result : int):
        """
            Add a comparison result between items[i] and items[j].
            result should be 1 if items[i] is better than items[j], else 0.
        """
        assert i != j, ValueError(f"Cannot add comparison for the same item {i} == {j}.")
        assert result in [0, 1], ValueError(f"Comparison result must be 0 or 1, got {result}.")
        if not self.is_compared(i, j):
            self._comparison_count += 1

        self.comparison_matrix[i][j] = result
        self.comparison_matrix[j][i] = 1 - result

    async def compare(self, i : int, j : int, reverse=False) -> bool:
        """
            Compare items[i] and items[j].
            Return True if items[i] is *smaller* than items[j] (for ascending order), else False.
            Args:
                reverse (`bool`) : If True, compare in descending order.
            Note:
                This function should be consistent at least, which means one and only one of a > b, a < b is True (assume no equality).
                The comparison may not be transitive, i.e., a > b and b > c does not necessarily imply a > c (refer to real-world scenarios).
        """
        assert i != j, ValueError(f"Cannot compare the same item {i} == {j}.")

        is_compared = self.is_compared(i, j)
        if not is_compared:
            if self.compare_fn is not None:
                win = await self.compare_fn(i, j)
            else:
                # Random comparison as placeholder
                win = np.random.choice([0, 1])

            self.add_comparison(i, j, win)

        return self.comparison_matrix[i][j] == (1 if reverse else 0)

    async def quick_sort(self, reverse=False, return_comparison_count=False):
        """
            Perform quicksort on self.items using pairwise comparisons.
            This algorithm *requires* the assumption that the comparison function is consistent and transitive.
        """
        cmp_cnt = 0
        arr = list(range(self.n))
        async def partition(low, high):
            pivot = arr[high]
            i = low - 1
            
            for j in range(low, high):
                nonlocal cmp_cnt
                cmp_cnt += 1
                if await self.compare(arr[j], pivot, reverse=reverse):
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
            
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            return i + 1
        
        async def quicksort_helper(low=0, high=None):
            if high is None:
                high = self.n - 1
            if low < high:
                pi = await partition(low, high)
                await quicksort_helper(low, pi - 1)
                await quicksort_helper(pi + 1, high)

        await quicksort_helper()
        if return_comparison_count:
            return [self.items[i] for i in arr], cmp_cnt
        
        return [self.items[i] for i in arr]

    
    async def tournament_sort(self, reverse=False, return_comparison_count=False):
        """
            Perform tournament sort on self.items using pairwise comparisons.
            This algorithm *does not* require the assumption that the comparison function is transitive.
        """
        cmp_cnt = 0
        arr = list(range(self.n))
        tmp = [0] * (2 * self.n)

        async def build_tournament():
            for i in range(self.n):
                tmp[self.n + i] = arr[i]
            for i in range(self.n - 1, 0, -1):
                left = tmp[2 * i]
                right = tmp[2 * i + 1]
                nonlocal cmp_cnt
                cmp_cnt += 1
                if await self.compare(left, right, reverse=reverse):
                    tmp[i] = left
                else:
                    tmp[i] = right

        async def update_tournament(winner_index):
            idx = (self.n + winner_index) // 2
            while idx > 0:
                left = tmp[2 * idx]
                right = tmp[2 * idx + 1]

                if left == -1:
                    tmp[idx] = right
                    idx //= 2
                    continue

                if right == -1:
                    tmp[idx] = left
                    idx //= 2
                    continue

                nonlocal cmp_cnt
                cmp_cnt += 1
                if await self.compare(left, right, reverse=reverse):
                    tmp[idx] = left
                else:
                    tmp[idx] = right
                idx //= 2

        sorted_items = []
        await build_tournament()
        for _ in range(self.n):
            winner = tmp[1]
            sorted_items.append(self.items[winner])
            winner_index = arr.index(winner)
            arr[winner_index] = -1  # Mark as removed
            tmp[self.n + winner_index] = -1  # Mark as removed in tournament tree
            await update_tournament(winner_index)

        if return_comparison_count:
            return sorted_items, cmp_cnt
        
        return sorted_items
    
    async def swiss_sort(self, reverse=False, return_comparison_count=False):
        """
            Perform Swiss-system tournament sort on self.items using pairwise comparisons.
            This algorithm *does not* require the assumption that the comparison function is transitive.
        """
        cmp_cnt = 0
        arr = list(range(self.n))
        scores = [0] * self.n
        rounds = int(np.ceil(np.log2(self.n))) + 1  # Number of rounds

        for _ in range(rounds):
            np.random.shuffle(arr)
            for i in range(0, self.n - 1, 2):
                cmp_cnt += 1
                if await self.compare(arr[i], arr[i + 1], reverse=reverse):
                    # If `compare` gives True, see it as arr[i] wins
                    scores[arr[i]] += 1
                else:
                    scores[arr[i + 1]] += 1

        sorted_indices = sorted(range(self.n), key=lambda x: scores[x], reverse=True)
        sorted_items = [self.items[i] for i in sorted_indices]

        if return_comparison_count:
            return sorted_items, cmp_cnt
        
        return sorted_items



# --------------------------------------------Preference Scorer-----------------------------------------------

def format_criteria(criteria : dict):
    s = ""
    for dimension, cri in criteria.items():
        s += f"### {dimension}"
        for sub_dim, sub_cri in cri.items():
            s += f"- {sub_dim}: {sub_cri}"
            s += "\n"
        
        s += '\n\n'

    return s

def build_messages(image1 : Image.Image, image2 : Image.Image, prompt: str, criteria=None, detailed=True):
    if criteria is None:
        criteria = ("- Theme Consistency (same topic or scenario across sub-images) "
                "- Style Consistency (colors, rendering style coherent) "
                "- Logical Consistency (actions or objects connect logically without contradictions). "
                "- Identify Consistency (the main subject should be consistently represented across sub-images). "
                "- Subject Consistency (main characters, animals, or objects remain the same across sub-images, "
                "including their identity, appearance, and fine-grained details). ")
    else:
        criteria = format_criteria(criteria)

    if detailed:
        question = "Answer with 'Yes' or 'No' and give detailed reasons."
    else:
        question = "Answer with 'Yes or 'No"

    messages = [
        {
            "role": "system",
            "content": (
                "You are an evaluator. Compare two images and decide whether the first image "
                "demonstrates better overall consistency than the second image. "
                "Consistency includes: \n" + criteria
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Image description: {prompt}"},
                {"type": "image_url", "image_url": {"url": pil_image_to_base64(image1)}},
                {"type": "image_url", "image_url": {"url": pil_image_to_base64(image2)}},
                {"type": "text", "text": f"Question: Does the first image have better consistency than the second image? {question}"}
            ],
        },
    ]
    return messages

class PrefScorer:
    def __init__(
            self,
            client: AsyncOpenAI,
            model='Qwen2.5-VL-7B-Instruct',
            max_concurrent=100,
            max_retries=10,
            timeout=60
        ):
        self.client = client
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout = timeout
        self.global_semaphore = asyncio.Semaphore(self.max_concurrent)

    def __call__(
        self, 
        images: List[Image.Image], 
        prompts: List[str], 
        metadata: List[dict], 
        detailed: bool = True,
        sort_algorithm: Literal['round_robin', 'quick_sort', 'tournament', 'swiss'] = 'round_robin',
        return_rankings: bool = False
    ) -> np.ndarray:
        """
        Score images based on preferences using different sorting algorithms.
        
        Args:
            images: List of PIL Images
            prompts: List of prompts corresponding to each image
            metadata: List of metadata dictionaries
            detailed: Whether to use detailed comparison prompts
            sort_algorithm: Which sorting algorithm to use, one of ['round_robin', 'quick_sort', 'tournament', 'swiss']
            return_rankings: If True, return rankings instead of scores
            
        Returns:
            np.ndarray: Scores or rankings for each image
        """
        return asyncio.run(self.__async_call__(images, prompts, metadata, detailed, sort_algorithm, return_rankings))

    async def __async_call__(
            self,
            images : list[Image.Image],
            prompts : list[str],
            metadata: list[dict],
            detailed: bool = True,
            sort_algorithm: Literal['round_robin', 'quick_sort', 'tournament', 'swiss'] = 'round_robin',
            return_rankings: bool = False
        ) -> np.ndarray:

        assert len(images) == len(prompts) == len(metadata), "Length of images, prompts, and metadata must be the same."
        # Group images and metadata by their prompts
        prompt_groups = {}
        for i, (img, prompt, meta) in enumerate(zip(images, prompts, metadata)):
            if prompt not in prompt_groups:
                prompt_groups[prompt] = {
                    'images': [],
                    'metadata': [],
                    'positions': []
                }
            prompt_groups[prompt]['images'].append(img)
            prompt_groups[prompt]['metadata'].append(meta)
            prompt_groups[prompt]['positions'].append(i)

        # Process each group concurrently
        all_scores = np.zeros(len(images), dtype=float)
        
        tasks = []
        for prompt, group_data in prompt_groups.items():
            task = self.compute_group_rewards(
                prompt, 
                group_data['images'], 
                group_data['metadata'][0],  # Use first metadata as representative
                sort_algorithm=sort_algorithm, 
                return_rankings=return_rankings,
                detailed=detailed
            )
            tasks.append((group_data['positions'], task))
        
        results = await asyncio.gather(*[t[1] for t in tasks])
        
        # Assign results back to all_scores
        for (positions, _), group_scores in zip(tasks, results):
            all_scores[positions] = group_scores

        return all_scores
        

    async def compute_group_rewards(
        self, 
        prompt: str, 
        images: List[Image.Image], 
        metadata: dict, 
        sort_algorithm: Literal['round_robin', 'quick_sort', 'tournament', 'swiss'] = 'round_robin',
        return_rankings: bool = False,
        detailed: bool = True
    ) -> np.ndarray:
        """Compute rewards for a group of images using the specified sorting algorithm."""
        n_images = len(images)
        
        if n_images <= 1:
            return np.array([1.0] if n_images == 1 else [])
        
        if sort_algorithm == 'round_robin':
            return await self._round_robin_scoring(images, prompt, metadata, detailed)
        else:
            return await self._efficient_sorting(images, prompt, metadata, sort_algorithm, return_rankings, detailed)

    async def _round_robin_scoring(self, images: List[Image.Image], prompt: str, metadata: dict, detailed=True) -> np.ndarray:
        """
            Original round-robin approach that compares all pairs.
        """
        n_images = len(images)
        comparison_matrix = np.zeros((n_images, n_images), dtype=np.float64)
        
        # Collect all pairs to compare
        tasks = []
        pairs = []
        
        for i, j in permutations(range(n_images), 2):
            task = self.compare_image(images[i], images[j], prompt, metadata, detailed=detailed)
            tasks.append(task)
            pairs.append((i, j))
        
        # Execute all comparisons concurrently
        completions = await asyncio.gather(*tasks)

        # Fill the comparison array
        for (i, j), completion in zip(pairs, completions):
            prob = get_yes_cond_prob_from_completion(completion, canonicalize=True)
            comparison_matrix[i, j] = prob

        # Convert probabilities to wins/losses
        for i, j in combinations(range(n_images), 2):
            win1 = int(comparison_matrix[i, j] > comparison_matrix[j, i])
            win2 = int(comparison_matrix[j, i] > comparison_matrix[i, j])
            comparison_matrix[i, j] = win1
            comparison_matrix[j, i] = win2

        # Calculate scores
        scores = comparison_matrix.sum(axis=1) / max(1, n_images - 1)
        return scores

    async def _efficient_sorting(
        self, 
        images: List[Image.Image], 
        prompt: str, 
        metadata: dict, 
        sort_algorithm: str,
        return_rankings: bool,
        detailed: bool = True
    ) -> np.ndarray:
        """Use efficient sorting algorithms with on-demand comparisons."""
        n_images = len(images)
        
        # Create async comparison function for this group
        async def async_compare_func(i: int, j: int) -> int:
            """
                Compare images[i] and images[j], return 1 if i is *better* (*smaller*), 0 otherwise.
            """
            completion1 = await self.compare_image(images[i], images[j], prompt, metadata, detailed=detailed)
            completion2 = await self.compare_image(images[j], images[i], prompt, metadata, detailed=detailed)
            prob1 = get_yes_cond_prob_from_completion(completion1, canonicalize=True)
            prob2 = get_yes_cond_prob_from_completion(completion2, canonicalize=True)
            return int(prob1 > prob2)
        
        # Create comparator with on-demand comparison capability
        comparator = GroupPairwiseComparator(
            items=list(range(n_images)), 
            compare_fn=async_compare_func
        )
        
        # Apply sorting algorithm
        if sort_algorithm == 'quick_sort':
            sorted_indices, cmp_count = await comparator.quick_sort(reverse=True, return_comparison_count=True)
        elif sort_algorithm == 'tournament':
            sorted_indices, cmp_count = await comparator.tournament_sort(reverse=True, return_comparison_count=True)
        elif sort_algorithm == 'swiss':
            sorted_indices, cmp_count = await comparator.swiss_sort(reverse=True, return_comparison_count=True)
        else:
            raise ValueError(f"Unknown sorting algorithm: {sort_algorithm}")
        
        if return_rankings:
            # Return rankings (0 = best, n-1 = worst)
            rankings = np.zeros(n_images, dtype=int)
            for rank, img_idx in enumerate(sorted_indices):
                rankings[img_idx] = rank
            return rankings.astype(float)
        else:
            # Return scores based on rankings (higher = better)
            scores = np.zeros(n_images, dtype=float)
            for rank, img_idx in enumerate(sorted_indices):
                scores[img_idx] = (n_images - rank - 1) / (n_images - 1)
            return scores

    async def compare_image(
            self,
            image1 : Image.Image,
            image2 : Image.Image,
            prompt : str = "",
            metadata : Optional[dict] = None,
            detailed : bool = False,
            top_logprobs: int = 20
        ) -> openai.ChatCompletion:
        if metadata is not None:
            criteria = metadata.get('criteria', None)
        else:
            criteria = None
        
        messages = build_messages(image1, image2, prompt, criteria, detailed=detailed)
        
        for attempt in range(self.max_retries):
            try:
                async with self.global_semaphore:
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
                    time.sleep(2 ** attempt)
                else:
                    completion = None

        return completion    


def main():
    reward_fn = pref_score()