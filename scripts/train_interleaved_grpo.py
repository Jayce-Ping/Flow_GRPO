"""
Interleaved Multi-Turn Dialogue RL Training for Bagel.

Combines DeepSeek's GRPO for text generation (turns 0..N-2) and Flow-GRPO
for image generation (turn N-1).  Each dialogue trajectory is scored as a
whole, and the shared advantage drives both losses with configurable weights.

Structure mirrors flow_grpo/scripts/train_bagel.py.
"""

from collections import defaultdict
import contextlib
import copy
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
import math
from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple

from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator, load_checkpoint_and_dispatch, init_empty_weights
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers.utils.torch_utils import is_compiled_module

# bagel
from flow_grpo.bagel.data.data_utils import add_special_tokens
from flow_grpo.bagel.data.transforms import ImageTransform
from flow_grpo.bagel.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from flow_grpo.bagel.modeling.qwen2 import Qwen2Tokenizer
from flow_grpo.bagel.modeling.autoencoder import load_ae
from flow_grpo.bagel.modeling.bagel.qwen2_navit import NaiveCache
from flow_grpo.bagel.inferencer import InterleaveInferencer
from flow_grpo.fsdp_utils import save_fsdp_checkpoint, register_optimizer_offload_hooks
from flow_grpo.stat_tracking import PerPromptStatTracker
import flow_grpo.rewards

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from huggingface_hub import snapshot_download


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/interleaved_grpo.py", "Training configuration.")
logger = get_logger(__name__)


# ──────────────────────────────── Dataset ────────────────────────────────

class InterleavedDialogueDataset(Dataset):
    """
    Each item is a multi-turn dialogue: turns 0..N-2 are text-to-text,
    turn N-1 is text-to-image.  Stored as JSONL with structure:
    {"turns": [{"role": "user", "content": "..."}, ...], "image_prompt": "..."}
    
    Alternatively, for simple text-to-image with preceding dialogue,
    a plain txt file where each line is a single image prompt (dialogue
    context will be sampled / constructed on the fly).
    """

    def __init__(self, dataset_path: str, split: str = "train"):
        self.file_path = os.path.join(dataset_path, f"{split}.jsonl")
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.data = [json.loads(line) for line in f]
        else:
            # Fallback: plain text prompts (one image-gen prompt per line)
            txt_path = os.path.join(dataset_path, f"{split}.txt")
            with open(txt_path, "r") as f:
                prompts = [l.strip() for l in f if l.strip()]
            if split == "test":
                prompts = prompts[:512]
            self.data = [{"turns": [], "image_prompt": p} for p in prompts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(examples):
        return examples


# ──────────────────────────── Distributed Sampler ────────────────────────

class DistributedKRepeatSampler(Sampler):
    """Yields batches where each prompt is repeated K times (for GRPO groups)."""

    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        n = len(self.dataset)
        indices = torch.randperm(n, generator=g).tolist()

        # Shard across replicas
        per_replica = math.ceil(n / self.num_replicas)
        start = self.rank * per_replica
        end = min(start + per_replica, n)
        indices = indices[start:end]

        # Repeat each index K times, then yield in batches
        repeated = []
        for idx in indices:
            repeated.extend([idx] * self.k)

        batch = []
        for idx in repeated:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self):
        per_replica = math.ceil(len(self.dataset) / self.num_replicas)
        return math.ceil(per_replica * self.k / self.batch_size)


# ────────────────────── Text-Generation GRPO Helpers ─────────────────────

@torch.no_grad()
def sample_text_with_logprobs(
    inferencer: InterleaveInferencer,
    gen_context: dict,
    max_length: int = 512,
    do_sample: bool = True,
    temperature: float = 0.7,
) -> Tuple[str, torch.Tensor, torch.Tensor, dict]:
    """
    Sample text autoregressively from the model and collect per-token log-probs.

    Returns:
        generated_text: The decoded string.
        token_ids: (seq_len,) tensor of generated token ids.
        log_probs: (seq_len,) tensor of per-token log-probs under the policy.
        updated_gen_context: KV-cache context after the generated text.
    """
    model = inferencer.model
    tokenizer = inferencer.tokenizer
    new_token_ids = inferencer.new_token_ids

    gen_context = deepcopy(gen_context)
    past_key_values = gen_context["past_key_values"]
    kv_lens = gen_context["kv_lens"]
    ropes = gen_context["ropes"]

    generation_input = model.prepare_start_tokens(kv_lens, ropes, new_token_ids)
    device = next(model.parameters()).device
    for k, v in generation_input.items():
        if torch.is_tensor(v):
            generation_input[k] = v.to(device)

    curr_tokens = generation_input["packed_start_tokens"]
    packed_query_position_ids = generation_input["packed_query_position_ids"]
    key_values_lens = generation_input["key_values_lens"]
    packed_key_value_indexes = generation_input["packed_key_value_indexes"]

    generated_ids = []
    token_log_probs = []
    step = 0
    eos_id = new_token_ids["eos_token_id"]

    while step < max_length:
        packed_text_embedding = model.language_model.model.embed_tokens(curr_tokens)
        query_lens = torch.ones_like(curr_tokens)
        packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
            0, len(key_values_lens), device=key_values_lens.device, dtype=key_values_lens.dtype
        )

        uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
        for i in range(len(uppacked)):
            uppacked[i] += i
        packed_key_value_indexes = torch.cat(uppacked, dim=0)

        extra_inputs = {}
        if model.use_moe:
            extra_inputs = {"mode": "und"}

        output = model.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        logits = model.language_model.lm_head(output.packed_query_sequence)  # (1, vocab)
        log_probs_all = F.log_softmax(logits / max(temperature, 1e-8), dim=-1)

        if do_sample and temperature > 0:
            probs = torch.exp(log_probs_all)
            next_token = torch.multinomial(probs.squeeze(0), num_samples=1)
        else:
            next_token = logits.argmax(dim=-1)

        next_token = next_token.view(-1)
        lp = log_probs_all.squeeze(0).gather(-1, next_token.unsqueeze(-1)).squeeze(-1)
        token_log_probs.append(lp)
        generated_ids.append(next_token)

        key_values_lens = key_values_lens + 1
        packed_key_value_indexes = torch.arange(
            key_values_lens.sum().item(), device=device, dtype=torch.long
        )
        packed_query_position_ids = packed_query_position_ids + 1
        curr_tokens = next_token

        step += 1
        if next_token.item() == eos_id:
            break

    token_ids = torch.cat(generated_ids, dim=0)
    log_probs_tensor = torch.cat(token_log_probs, dim=0)

    text = tokenizer.decode(token_ids.cpu().tolist())
    text = text.split("<|im_end|>")[0]
    if "<|im_start|>" in text:
        text = text.split("<|im_start|>")[1]

    # Update gen_context
    gen_context["past_key_values"] = past_key_values
    gen_context["kv_lens"] = [kv_lens[0] + len(generated_ids)]
    gen_context["ropes"] = [ropes[0] + len(generated_ids)]

    return text, token_ids, log_probs_tensor, gen_context


def compute_text_grpo_loss(
    model,
    ref_model,
    token_ids: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: float,
    gen_context_before: dict,
    new_token_ids: dict,
    clip_range: float = 0.2,
    beta: float = 0.01,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute GRPO policy-gradient loss for text tokens.

    Uses the standard PPO-clip objective with an optional KL penalty
    against the reference model (DeepSeek GRPO style).

    Args:
        model: The trainable Bagel model.
        ref_model: Frozen reference LM for KL penalty.
        token_ids: (T,) generated token ids.
        old_log_probs: (T,) log-probs from the sampling policy.
        advantages: Scalar advantage for this trajectory.
        gen_context_before: KV-cache state *before* text was generated.
        new_token_ids: Special token id dict.
        clip_range: PPO clipping epsilon.
        beta: KL penalty coefficient.

    Returns:
        loss: Scalar loss tensor.
        info: Dict with diagnostic values.
    """
    device = next(model.parameters()).device
    T = token_ids.shape[0]
    if T == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {}

    # Recompute log-probs under the current policy (with gradients)
    past_kv = gen_context_before["past_key_values"]
    kv_lens = gen_context_before["kv_lens"]
    ropes = gen_context_before["ropes"]

    generation_input = model.prepare_start_tokens(kv_lens, ropes, new_token_ids)
    for k, v in generation_input.items():
        if torch.is_tensor(v):
            generation_input[k] = v.to(device)

    # Build the full input: [bos_token, token_ids[0], ..., token_ids[T-2]]
    # so that position i predicts token_ids[i]
    input_tokens = torch.cat([
        generation_input["packed_start_tokens"],
        token_ids[:-1],
    ], dim=0)

    packed_text_embedding = model.language_model.model.embed_tokens(input_tokens)
    query_lens = torch.tensor([T], device=device, dtype=torch.int)

    # Build position ids and indexes for a single causal sequence
    base_kv_len = generation_input["key_values_lens"].sum().item()
    packed_query_position_ids = generation_input["packed_query_position_ids"][0] + torch.arange(
        T, device=device, dtype=torch.long
    )
    packed_query_indexes = base_kv_len + torch.arange(T, device=device, dtype=torch.long)
    packed_key_value_indexes = generation_input["packed_key_value_indexes"]

    extra_inputs = {}
    if model.use_moe:
        extra_inputs = {"mode": "und"}

    output = model.language_model.forward_inference(
        packed_query_sequence=packed_text_embedding,
        query_lens=query_lens,
        packed_query_position_ids=packed_query_position_ids,
        packed_query_indexes=packed_query_indexes,
        past_key_values=past_kv,
        packed_key_value_indexes=packed_key_value_indexes,
        key_values_lens=generation_input["key_values_lens"],
        update_past_key_values=False,
        is_causal=True,
        **extra_inputs,
    )

    logits = model.language_model.lm_head(output.packed_query_sequence)  # (T, vocab)
    new_log_probs = F.log_softmax(logits, dim=-1)
    new_lp = new_log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # (T,)

    # Reference model log-probs (no gradient)
    ref_lp = torch.zeros_like(new_lp)
    if ref_model is not None and beta > 0:
        with torch.no_grad():
            ref_embedding = ref_model.model.embed_tokens(input_tokens)
            ref_output = ref_model.forward_inference(
                packed_query_sequence=ref_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=deepcopy(past_kv),
                packed_key_value_indexes=packed_key_value_indexes,
                key_values_lens=generation_input["key_values_lens"],
                update_past_key_values=False,
                is_causal=True,
                **extra_inputs,
            )
            ref_logits = ref_model.lm_head(ref_output.packed_query_sequence)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_lp = ref_log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

    # PPO-clip objective (per token, then mean)
    ratio = torch.exp(new_lp - old_log_probs.detach())
    adv = torch.tensor(advantages, device=device, dtype=ratio.dtype)
    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    policy_loss = torch.maximum(unclipped, clipped).mean()

    # KL penalty
    kl_loss = torch.tensor(0.0, device=device)
    if beta > 0 and ref_model is not None:
        kl_loss = (new_lp - ref_lp).mean()  # Approximate KL

    loss = policy_loss + beta * kl_loss

    info = {
        "text_policy_loss": policy_loss.detach(),
        "text_kl_loss": kl_loss.detach(),
        "text_loss": loss.detach(),
        "text_ratio_mean": ratio.mean().detach(),
    }
    return loss, info


# ────────────────────────── Utility Functions ────────────────────────────

def create_generators(prompts, base_seed):
    """Create deterministic torch.Generators per prompt for reproducible sampling."""
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], "big")
        seed = (base_seed + prompt_hash_int) % (2 ** 31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


def calculate_zero_std_ratio(prompts, gathered_rewards):
    """Compute the ratio of prompt groups with zero reward std."""
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped = gathered_rewards["ori_avg"][np.argsort(inverse_indices)]
    splits = np.cumsum(counts)[:-1]
    groups = np.split(grouped, splits)
    stds = np.array([np.std(g) for g in groups])
    zero_ratio = np.count_nonzero(stds == 0) / len(stds)
    return zero_ratio, stds.mean()


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


# ──────────────────── Interleaved Sampling Pipeline ──────────────────────

@torch.no_grad()
def sample_interleaved_trajectory(
    inferencer: InterleaveInferencer,
    dialogue: dict,
    config,
    accelerator,
    inference_hyper: dict,
    generator=None,
) -> dict:
    """
    Sample a full interleaved trajectory:
      - For each text turn: generate text via autoregressive sampling
        and record token-level log-probs.
      - For the final turn: generate an image via flow-matching
        and record latent-level log-probs.

    Args:
        inferencer: The InterleaveInferencer wrapping the Bagel model.
        dialogue: Dict with "turns" (list of user messages) and "image_prompt".
        config: Training config.
        accelerator: Accelerate instance.
        inference_hyper: Image-generation hyperparameters.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        Dict containing all info needed for training:
          - text_turns: list of dicts with token_ids, log_probs, gen_context_before
          - image_data: dict with latents, log_probs, timesteps, image tensor
          - image: generated image tensor (C, H, W)
          - prompt_text: the full dialogue text for reward scoring
    """
    gen_context = inferencer.init_gen_context()
    cfg_text_context = deepcopy(gen_context)
    cfg_img_context = deepcopy(gen_context)

    text_turn_data = []
    all_text_parts = []

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        # ── Process user turns (text context) ──
        turns = dialogue.get("turns", [])
        for turn in turns:
            user_msg = turn["content"] if isinstance(turn, dict) else turn
            # Update context with user message
            cfg_text_context = deepcopy(gen_context)
            gen_context = inferencer.update_context_text(user_msg, gen_context)
            cfg_img_context = inferencer.update_context_text(user_msg, cfg_img_context)
            all_text_parts.append(f"User: {user_msg}")

            # Generate assistant text response
            gen_context_before_text = deepcopy(gen_context)
            text, token_ids, log_probs, gen_context = sample_text_with_logprobs(
                inferencer, gen_context,
                max_length=config.sample.get("max_text_length", 512),
                do_sample=True,
                temperature=config.sample.get("text_temperature", 0.7),
            )

            # Also update cfg contexts with generated text
            gen_context = inferencer.update_context_text(text, deepcopy(gen_context_before_text))
            cfg_img_context = inferencer.update_context_text(text, cfg_img_context)
            all_text_parts.append(f"Assistant: {text}")

            text_turn_data.append({
                "text": text,
                "token_ids": token_ids,
                "log_probs": log_probs,
                "gen_context_before": gen_context_before_text,
            })

        # ── Final turn: image generation ──
        image_prompt = dialogue["image_prompt"]
        cfg_text_context = deepcopy(gen_context)
        gen_context = inferencer.update_context_text(image_prompt, gen_context)
        cfg_img_context = inferencer.update_context_text(image_prompt, cfg_img_context)
        all_text_parts.append(f"User: {image_prompt}")

        image_shapes = inference_hyper.get("image_shapes", (512, 512))
        generators = [generator] if generator is not None else None

        # Sample image with log-probs via Flow-GRPO
        img_result = inferencer.gen_image(
            image_shapes,
            gen_context,
            cfg_text_precontext=cfg_text_context,
            cfg_img_precontext=cfg_img_context,
            cfg_text_scale=inference_hyper.get("cfg_text_scale", 4.0),
            cfg_img_scale=inference_hyper.get("cfg_img_scale", 1.0),
            cfg_interval=inference_hyper.get("cfg_interval", [0, 1.0]),
            timestep_shift=inference_hyper.get("timestep_shift", 3.0),
            num_timesteps=config.sample.num_steps,
            cfg_renorm_min=inference_hyper.get("cfg_renorm_min", 0.0),
            cfg_renorm_type=inference_hyper.get("cfg_renorm_type", "global"),
            noise_level=config.sample.noise_level,
            generators=generators,
            learn=False,  # sampling mode
        )

    return {
        "text_turns": text_turn_data,
        "image_data": {
            "all_latents": img_result.get("all_latents", []),
            "all_log_probs": img_result.get("all_log_probs", []),
            "timesteps": img_result.get("timesteps", torch.tensor([])),
        },
        "image": img_result["image"],
        "prompt_text": "\n".join(all_text_parts),
        "image_prompt": image_prompt,
        "dialogue": dialogue,
    }


# ──────────────────── Training Step (Combined Loss) ──────────────────────

def train_interleaved_step(
    trajectory: dict,
    advantage: float,
    inferencer: InterleaveInferencer,
    ref_model,
    config,
    accelerator,
    optimizer,
    transformer,
    inference_hyper: dict,
) -> dict:
    """
    Compute combined GRPO loss for a single trajectory.

    For text turns: standard GRPO with PPO-clip.
    For image turn: Flow-GRPO (handled by model.generate_image_learn).

    Returns:
        Dict of training metrics.
    """
    model = inferencer.model
    new_token_ids = inferencer.new_token_ids
    info = {}

    text_loss_weight = config.train.get("text_loss_weight", 1.0)
    image_loss_weight = config.train.get("image_loss_weight", 1.0)
    total_loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)

    # ── Text GRPO losses ──
    text_losses = []
    for turn_data in trajectory["text_turns"]:
        t_loss, t_info = compute_text_grpo_loss(
            model=model,
            ref_model=ref_model,
            token_ids=turn_data["token_ids"].to(accelerator.device),
            old_log_probs=turn_data["log_probs"].to(accelerator.device),
            advantages=advantage,
            gen_context_before=turn_data["gen_context_before"],
            new_token_ids=new_token_ids,
            clip_range=config.train.get("text_clip_range", 0.2),
            beta=config.train.get("text_beta", 0.01),
        )
        text_losses.append(t_loss)
        for k, v in t_info.items():
            info.setdefault(k, []).append(v)

    if text_losses:
        avg_text_loss = torch.stack(text_losses).mean()
        total_loss = total_loss + text_loss_weight * avg_text_loss
        info["avg_text_loss"] = avg_text_loss.detach()

    # ── Image Flow-GRPO loss ──
    image_data = trajectory["image_data"]
    if len(image_data["all_latents"]) > 0:
        # Reconstruct the flow-grpo sample dict expected by generate_image_learn
        latents_list = image_data["all_latents"]
        log_probs_list = image_data["all_log_probs"]
        timesteps = image_data["timesteps"]

        # Build sample dict matching existing flow_grpo format
        flow_sample = {
            "latents": latents_list[:-1] if len(latents_list) > 1 else latents_list,
            "prev_latents": latents_list[1:] if len(latents_list) > 1 else latents_list,
            "log_probs": log_probs_list,
            "timesteps": timesteps,
            "advantages": torch.tensor(advantage, device=accelerator.device),
        }

        # Re-run interleaved inference in learn mode for the image turn
        dialogue = trajectory["dialogue"]
        image_prompt = trajectory["image_prompt"]

        gen_context = inferencer.init_gen_context()
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # Replay text context (no grad needed for context building)
            turns = dialogue.get("turns", [])
            for i, turn in enumerate(turns):
                user_msg = turn["content"] if isinstance(turn, dict) else turn
                gen_context = inferencer.update_context_text(user_msg, gen_context)
                cfg_img_context = inferencer.update_context_text(user_msg, cfg_img_context)
                # Replay generated text
                if i < len(trajectory["text_turns"]):
                    gen_text = trajectory["text_turns"][i]["text"]
                    gen_context = inferencer.update_context_text(gen_text, gen_context)
                    cfg_img_context = inferencer.update_context_text(gen_text, cfg_img_context)

            cfg_text_context = deepcopy(gen_context)
            gen_context = inferencer.update_context_text(image_prompt, gen_context)
            cfg_img_context = inferencer.update_context_text(image_prompt, cfg_img_context)

            # Now call gen_image in learn mode
            img_output = inferencer.gen_image(
                inference_hyper.get("image_shapes", (512, 512)),
                gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=inference_hyper.get("cfg_text_scale", 4.0),
                cfg_img_scale=inference_hyper.get("cfg_img_scale", 1.0),
                cfg_interval=inference_hyper.get("cfg_interval", [0, 1.0]),
                timestep_shift=inference_hyper.get("timestep_shift", 3.0),
                num_timesteps=config.sample.num_steps,
                cfg_renorm_min=inference_hyper.get("cfg_renorm_min", 0.0),
                cfg_renorm_type=inference_hyper.get("cfg_renorm_type", "global"),
                noise_level=config.sample.noise_level,
                learn=True,
                sample=flow_sample,
                grpo_config=config,
                accelerator=accelerator,
                optimizer=optimizer,
                transformer=transformer,
            )

        if isinstance(img_output, dict):
            img_loss = img_output.get("loss", torch.tensor(0.0, device=accelerator.device))
            total_loss = total_loss + image_loss_weight * img_loss
            info["img_policy_loss"] = img_output.get("policy_loss", torch.tensor(0.0)).detach()
            info["img_kl_loss"] = img_output.get("kl_loss", torch.tensor(0.0)).detach()
            info["img_loss"] = img_loss.detach()
            info["img_clipfrac"] = img_output.get("clipfrac", torch.tensor(0.0)).detach()

    info["total_loss"] = total_loss.detach()

    # Backward (if not already done inside generate_image_learn)
    # Note: flow_grpo's generate_image_learn does backward internally per timestep.
    # We only need backward for the text loss portion.
    if text_losses:
        accelerator.backward(text_loss_weight * avg_text_loss)

    return info


# ──────────────────────────── Eval Function ──────────────────────────────

@torch.no_grad()
def eval_interleaved(
    inferencer, inference_hyper, test_dataloader, tokenizer,
    config, accelerator, global_step, eval_reward_fn, executor, autocast,
):
    """Evaluate by sampling trajectories and computing rewards."""
    all_rewards = defaultdict(list)

    for batch in tqdm(
        test_dataloader,
        desc="Eval",
        disable=not accelerator.is_local_main_process,
    ):
        images = []
        prompts = []
        with autocast():
            for dialogue in batch:
                traj = sample_interleaved_trajectory(
                    inferencer, dialogue, config, accelerator, inference_hyper
                )
                images.append(traj["image"])
                prompts.append(traj["image_prompt"])

        images_tensor = torch.stack(images, dim=0)
        rewards, _ = eval_reward_fn(images_tensor, prompts, [{}] * len(prompts))

        for key, value in rewards.items():
            gathered = accelerator.gather(
                torch.as_tensor(value, device=accelerator.device)
            ).cpu().numpy()
            all_rewards[key].append(gathered)

    all_rewards = {k: np.concatenate(v) for k, v in all_rewards.items()}
    if accelerator.is_main_process:
        wandb.log(
            {f"eval_reward_{k}": np.mean(v) for k, v in all_rewards.items()},
            step=global_step,
        )


# ──────────────────────────── Main Training ──────────────────────────────

def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = (config.run_name or "") + "_" + unique_id

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=(
            config.train.gradient_accumulation_steps
            * config.sample.train_batch_size
            * config.sample.get("sde_window_size", 1)
        ),
    )
    accelerator.state.fsdp_plugin.activation_checkpointing = config.get(
        "activation_checkpointing", False
    )
    accelerator.state.fsdp_plugin.transformer_cls_names_to_wrap = ["Qwen2MoTDecoderLayer"]

    if accelerator.is_main_process:
        wandb.init(project="interleaved_grpo", name=config.run_name, config=config.to_dict())
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    # ── Model Setup ──
    inference_dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float16

    model_path = config.pretrained.model
    model_local_dir = (
        snapshot_download(repo_id=model_path)
        if not os.path.exists(model_path)
        else model_path
    )

    llm_config = Qwen2Config.from_json_file(os.path.join(model_local_dir, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vae_model, vae_config = load_ae(local_path=os.path.join(model_local_dir, "ae.safetensors"))

    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=False,
        llm_config=llm_config,
        vit_config=None,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        model = Bagel(language_model, None, bagel_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_local_dir)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(512, 256, 8)
    vit_transform = ImageTransform(490, 112, 7)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_local_dir, "ema.safetensors"),
        device_map={"": f"cuda:{accelerator.local_process_index}"},
        offload_buffers=False,
        dtype=inference_dtype,
        force_hooks=True,
        offload_folder="/tmp/offload",
    )
    model = model.eval()

    # Reference model for KL penalty
    ref_model = None
    if config.train.get("beta", 0) > 0 or config.train.get("text_beta", 0) > 0:
        ref_model = Qwen2ForCausalLM(llm_config)
        ref_model.load_state_dict(model.language_model.state_dict())
        ref_model.to(device=f"cuda:{accelerator.local_process_index}", dtype=inference_dtype)
        ref_model.eval()
        ref_model.requires_grad_(False)

    vae_model.requires_grad_(False)
    model.requires_grad_(False)

    # ── LoRA or full fine-tuning ──
    if config.get("use_lora", False):
        # Set correct lora layers
        target_modules = [
            "self_attn.q_proj_moe_gen",
            "self_attn.k_proj_moe_gen",
            "self_attn.v_proj_moe_gen",
            "self_attn.o_proj_moe_gen",
            "mlp_moe_gen.gate_proj",
            "mlp_moe_gen.up_proj",
            "mlp_moe_gen.down_proj",
        ]
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        model.language_model = get_peft_model(model.language_model, transformer_lora_config)
        for name, param in model.language_model.named_parameters():
            if 'lora' in name:
                param.data = param.data.to(dtype=inference_dtype)
    else:
        for name, param in model.language_model.named_parameters():
            if "moe_gen" in name:
                param.requires_grad = True

    transformer = model.language_model
    transformer.config.use_cache = False
    trainable_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    if config.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    if config.get("fsdp_optimizer_offload", False):
        register_optimizer_offload_hooks(optimizer)

    # ── Data ──
    train_dataset = InterleavedDialogueDataset(config.dataset, "train")
    test_dataset = InterleavedDialogueDataset(config.dataset, "test")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.get("num_image_per_prompt", 4),
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=InterleavedDialogueDataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.get("test_batch_size", 4),
        collate_fn=InterleavedDialogueDataset.collate_fn,
        shuffle=False,
    )

    # ── Inferencer ──
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    inference_hyper = dict(
        cfg_text_scale=config.sample.get("guidance_scale", 4.0),
        cfg_img_scale=1.0,
        cfg_interval=[0, 1.0],
        timestep_shift=3.0,
        num_timesteps=config.sample.num_steps,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=tuple(config.get("image_shapes", [512, 512])),
    )

    # ── Reward function ──
    reward_fn = getattr(flow_grpo.rewards, config.reward_fn)(accelerator.device)
    eval_reward_fn = reward_fn  # Can be different if needed
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # ── Per-prompt stat tracker for advantage normalization ──
    stat_tracker = PerPromptStatTracker() if config.get("per_prompt_stat_tracking", True) else None

    # ── FSDP Wrap ──
    transformer = accelerator.prepare(transformer)
    optimizer = accelerator.prepare(optimizer)

    autocast = contextlib.partial(accelerator.autocast) if hasattr(accelerator, "autocast") else contextlib.nullcontext

    # ── Training Loop ──
    train_iter = iter(train_dataloader)
    global_step = 0
    first_epoch = 0

    if config.get("resume_from", ""):
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1

    for epoch in range(first_epoch, config.num_epochs):

        # ── Checkpoint & Eval ──
        if not config.get("debug", False) and epoch % config.get("save_freq", 10) == 0 and epoch > 0:
            save_fsdp_checkpoint(config.save_dir, transformer, global_step, accelerator.process_index)
        if not config.get("debug", False) and epoch % config.get("eval_freq", 5) == 0 and epoch > 0:
            transformer.eval()
            eval_interleaved(
                inferencer, inference_hyper, test_dataloader, tokenizer,
                config, accelerator, global_step, eval_reward_fn, executor, autocast,
            )

        # ══════════════════════ SAMPLING ══════════════════════
        transformer.eval()
        all_trajectories = []

        for batch_idx in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + batch_idx)
            batch = next(train_iter)

            with torch.no_grad():
                for dialogue in batch:
                    generator = None
                    if config.sample.get("same_latent", False):
                        prompt_key = dialogue.get("image_prompt", str(dialogue))
                        generator = create_generators([prompt_key], base_seed=42)[0]

                    traj = sample_interleaved_trajectory(
                        inferencer, dialogue, config, accelerator, inference_hyper, generator
                    )
                    all_trajectories.append(traj)

        # ── Compute Rewards ──
        images = torch.stack([t["image"] for t in all_trajectories], dim=0)
        image_prompts = [t["image_prompt"] for t in all_trajectories]
        metadatas = [{}] * len(all_trajectories)

        rewards_future = executor.submit(reward_fn, images, image_prompts, metadatas)
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()

        reward_tensor = torch.as_tensor(rewards["avg"], device=accelerator.device).float()

        # Log rewards
        gathered_rewards = accelerator.gather(reward_tensor).cpu().numpy()
        if accelerator.is_main_process:
            wandb.log({"epoch": epoch, "reward_avg": gathered_rewards.mean()}, step=global_step)

        # ── Compute Advantages ──
        if stat_tracker is not None:
            prompt_ids_encoded = tokenizer(
                image_prompts, padding="max_length", max_length=256,
                truncation=True, return_tensors="pt",
            ).input_ids.to(accelerator.device)
            gathered_prompt_ids = accelerator.gather(prompt_ids_encoded).cpu().numpy()
            all_prompts = tokenizer.batch_decode(gathered_prompt_ids, skip_special_tokens=True)
            advantages = stat_tracker.update(all_prompts, gathered_rewards)
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards - gathered_rewards.mean()) / (gathered_rewards.std() + 1e-4)

        # Map back to local process
        advantages = torch.as_tensor(advantages, device=accelerator.device)
        local_advantages = advantages.reshape(
            accelerator.num_processes, -1
        )[accelerator.process_index]

        # ── Log sample images ──
        if epoch % 5 == 0 and accelerator.is_main_process:
            with tempfile.TemporaryDirectory() as tmpdir:
                n_log = min(8, len(images))
                for idx in range(n_log):
                    img = images[idx]
                    pil = Image.fromarray(
                        (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
                wandb.log(
                    {
                        "samples": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{image_prompts[idx][:100]} | r={reward_tensor[idx]:.2f}",
                            )
                            for idx in range(n_log)
                        ]
                    },
                    step=global_step,
                )

        # ══════════════════════ TRAINING ══════════════════════
        # Set False to use `forward_inference`
        transformer.train()
        transformer.module.training = False
        transformer.module.model.training = False
        if config.use_lora:
            transformer.module.model.model.training = False
            for layer in transformer.module.model.model.layers:
                layer.module.training = False
                layer.module.self_attn.training = False
        else:
            for layer in transformer.module.model.layers:
                layer.module.training = False
                layer.module.self_attn.training = False

        for inner_epoch in range(config.train.num_inner_epochs):
            info = defaultdict(list)

            for i, traj in tqdm(
                list(enumerate(all_trajectories)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                disable=not accelerator.is_local_main_process,
            ):
                adv = local_advantages[i].item()
                adv = max(min(adv, config.train.get("adv_clip_max", 5.0)),
                          -config.train.get("adv_clip_max", 5.0))

                with accelerator.accumulate(transformer):
                    step_info = train_interleaved_step(
                        trajectory=traj,
                        advantage=adv,
                        inferencer=inferencer,
                        ref_model=ref_model,
                        config=config,
                        accelerator=accelerator,
                        optimizer=optimizer,
                        transformer=transformer,
                        inference_hyper=inference_hyper,
                    )

                    for k, v in step_info.items():
                        if isinstance(v, (list, torch.Tensor)):
                            info[k].append(
                                v.mean() if isinstance(v, list) and isinstance(v[0], torch.Tensor)
                                else (v if isinstance(v, torch.Tensor) else torch.tensor(v))
                            )
                        elif isinstance(v, (int, float)):
                            info[k].append(torch.tensor(v))

                    if accelerator.sync_gradients:
                        torch.nn.utils.clip_grad_norm_(trainable_params, config.train.get("max_grad_norm", 1.0))
                        optimizer.step()
                        optimizer.zero_grad()

                if accelerator.sync_gradients:
                    log_info = {k: torch.stack(v).mean().item() for k, v in info.items() if v}
                    log_info["epoch"] = epoch
                    log_info["inner_epoch"] = inner_epoch
                    if accelerator.is_main_process:
                        wandb.log(log_info, step=global_step)
                    global_step += 1
                    info = defaultdict(list)

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)