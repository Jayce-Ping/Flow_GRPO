"""
data_process.py — Convert raw multi-stage pipeline outputs to training JSONL.

Raw data layout (from screenshots & trajectory JSON):
  stage2_dir/                          (apdcephfs_zwfy2/share_303944931/...)
    ├── {ip_index}_trajectory.json     # full multi-turn trajectory
    └── intermediate/
        └── {ip_index}/
            ├── image_1.jpg            # best reference image (from search)
            └── image_2.jpg            # 2nd reference image

  stage3_dir/                          (stage3_generation/final_search_gemini3_v2)
    └── {ip_index}_{...}_0.png         # generated image (filename starts with ip_index)

Output (for InterleavedDialogueDataset):
  output_dir/
    ├── train.jsonl                    # {"turns": [...], "image_prompt": "...", "ref_images": [...]}
    ├── test.jsonl
    └── images/                        # symlinked or copied reference images
        └── {ip_index}/
            ├── image_1.jpg
            └── image_2.jpg

Each JSONL line has the structure expected by InterleavedDialogueDataset:
  {
    "turns": [
      {"role": "user", "content": "<stage1 initial prompt>"},
      {"role": "user", "content": "<stage2 image search prompt>"}
    ],
    "image_prompt": "<recaption from stage3>",
    "ref_images": ["images/{ip_index}/image_1.jpg", ...],
    "metadata": {
      "ip_index": "...", "ip_name": "...", "language": "...",
      "original_image_prompt": "...", "generated_image": "..."
    }
  }
"""

import os
import re
import json
import glob
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ─────────────────────── Discovery helpers ────────────────────────────────

def discover_trajectories(stage2_dir: str) -> Dict[str, str]:
    """Find all *_trajectory.json files.  Returns {ip_index: json_path}."""
    pattern = os.path.join(stage2_dir, "*_trajectory.json")
    results = {}
    for path in glob.glob(pattern):
        basename = os.path.basename(path)
        ip_index = basename.replace("_trajectory.json", "")
        results[ip_index] = path
    return results


def discover_reference_images(stage2_dir: str, ip_index: str) -> List[str]:
    """Find reference images under intermediate/{ip_index}/."""
    img_dir = os.path.join(stage2_dir, "intermediate", ip_index)
    if not os.path.isdir(img_dir):
        return []
    imgs = []
    for name in sorted(os.listdir(img_dir)):
        if name.startswith("image_") and name.lower().endswith((".jpg", ".jpeg", ".png")):
            imgs.append(os.path.join(img_dir, name))
    return imgs


def discover_generated_image(stage3_dir: str, ip_index: str) -> Optional[str]:
    """Find generated image in stage3 dir.  Filename starts with ip_index."""
    if not os.path.isdir(stage3_dir):
        return None
    for name in os.listdir(stage3_dir):
        if name.startswith(ip_index) and name.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(stage3_dir, name)
    return None


# ─────────────────────── Trajectory parsing ───────────────────────────────

def extract_recaption(text: str) -> str:
    """Extract content inside <recaption>...</recaption> tags."""
    if not text:
        return ""
    m = re.search(r"<recaption>(.*?)</recaption>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"<recaption>(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def parse_trajectory(traj_path: str) -> Optional[Dict]:
    """Parse a trajectory JSON into a structured dict.

    Returns None if the trajectory is invalid / incomplete.
    """
    with open(traj_path, "r", encoding="utf-8") as f:
        traj = json.load(f)

    ip_index = traj.get("ip_index", "")
    ip_name = traj.get("ip_name", "")
    image_prompt = traj.get("image_prompt", "")
    language = traj.get("language", "en")
    country = traj.get("country", "")

    # Extract recaption (the detailed prompt for image generation)
    recaption = traj.get("recaption", "")
    if not recaption:
        # Try to find it in the last turn's response
        turns = traj.get("turns", [])
        for turn in reversed(turns):
            resp = turn.get("response_text", "")
            rc = extract_recaption(resp)
            if rc:
                recaption = rc
                break

    if not recaption:
        return None  # No usable recaption → skip

    # Build dialogue turns for training
    # From the trajectory, stage1 and stage2 are text-to-text dialogue turns;
    # The recaption becomes the final image_prompt.
    dialogue_turns = []
    turns = traj.get("turns", [])

    for turn in turns:
        stage = turn.get("stage", 0)
        if stage == 3:
            continue  # stage3 is image generation, not a text turn

        # User message is the 'input' field
        user_input = turn.get("input", "")
        if not user_input:
            continue

        # Assistant response
        response = turn.get("response_text", "")

        # Tool output (becomes the observation injected between turns)
        tool_output = turn.get("tool_output", "")

        dialogue_turns.append({
            "role": "user",
            "content": user_input,
            "response": response,
            "tool_output": tool_output if tool_output else None,
        })

    return {
        "ip_index": ip_index,
        "ip_name": ip_name,
        "language": language,
        "country": country,
        "original_image_prompt": image_prompt,
        "recaption": recaption,
        "dialogue_turns": dialogue_turns,
    }


# ─────────────────────── Format converters ────────────────────────────────

def build_simple_training_sample(parsed: Dict, ref_image_relpaths: List[str],
                                  gen_image_relpath: Optional[str]) -> Dict:
    """Build a *simple* training sample (minimal text turns + image_prompt).

    This matches InterleavedDialogueDataset expected format:
      {"turns": [...], "image_prompt": "...", ...}
    """
    turns = []
    for dt in parsed["dialogue_turns"]:
        turns.append({"role": "user", "content": dt["content"]})

    return {
        "turns": turns,
        "image_prompt": parsed["recaption"],
        "ref_images": ref_image_relpaths,
        "metadata": {
            "ip_index": parsed["ip_index"],
            "ip_name": parsed["ip_name"],
            "language": parsed["language"],
            "original_image_prompt": parsed["original_image_prompt"],
            "generated_image": gen_image_relpath,
        },
    }


def build_full_training_sample(parsed: Dict, ref_image_relpaths: List[str],
                                gen_image_relpath: Optional[str]) -> Dict:
    """Build a *full* training sample preserving assistant responses & tool outputs.

    This is richer and allows training on the full multi-turn interaction including
    the assistant's text_search / search_image tool use reasoning.
    """
    turns = []
    for dt in parsed["dialogue_turns"]:
        turns.append({"role": "user", "content": dt["content"]})
        if dt.get("response"):
            turns.append({"role": "assistant", "content": dt["response"]})
        if dt.get("tool_output"):
            turns.append({"role": "observation", "content": dt["tool_output"]})

    return {
        "turns": turns,
        "image_prompt": parsed["recaption"],
        "ref_images": ref_image_relpaths,
        "metadata": {
            "ip_index": parsed["ip_index"],
            "ip_name": parsed["ip_name"],
            "language": parsed["language"],
            "original_image_prompt": parsed["original_image_prompt"],
            "generated_image": gen_image_relpath,
        },
    }


# ─────────────────────── Main processing ──────────────────────────────────

def process_dataset(
    stage2_dir: str,
    stage3_dir: str,
    output_dir: str,
    test_ratio: float = 0.05,
    copy_images: bool = False,
    full_turns: bool = False,
    min_recaption_len: int = 50,
    seed: int = 42,
):
    """Process raw stage2 + stage3 data into training JSONL.

    Args:
        stage2_dir: Root dir containing *_trajectory.json + intermediate/.
        stage3_dir: Dir containing generated images ({ip_index}_*.png).
        output_dir: Where to write train.jsonl, test.jsonl, images/.
        test_ratio: Fraction of data for test split.
        copy_images: If True, copy images; otherwise create symlinks.
        full_turns: If True, include assistant responses & tool outputs in turns.
        min_recaption_len: Skip samples with recaption shorter than this.
        seed: Random seed for train/test split.
    """
    import random
    random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # 1. Discover all trajectories
    print(f"📂 Scanning stage2 dir: {stage2_dir}")
    traj_map = discover_trajectories(stage2_dir)
    print(f"   Found {len(traj_map)} trajectory files")

    # 2. Process each trajectory
    samples = []
    stats = defaultdict(int)

    for ip_index, traj_path in sorted(traj_map.items()):
        stats["total"] += 1

        # Parse trajectory
        parsed = parse_trajectory(traj_path)
        if parsed is None:
            stats["skip_no_recaption"] += 1
            continue

        if len(parsed["recaption"]) < min_recaption_len:
            stats["skip_short_recaption"] += 1
            continue

        # Find reference images
        ref_images = discover_reference_images(stage2_dir, ip_index)
        if not ref_images:
            stats["skip_no_ref_images"] += 1
            continue

        # Find generated image
        gen_image = discover_generated_image(stage3_dir, ip_index)

        # Copy/link reference images to output
        ip_img_dir = os.path.join(images_dir, ip_index)
        os.makedirs(ip_img_dir, exist_ok=True)

        ref_relpaths = []
        for src_path in ref_images:
            fname = os.path.basename(src_path)
            dst_path = os.path.join(ip_img_dir, fname)
            relpath = os.path.join("images", ip_index, fname)

            if not os.path.exists(dst_path):
                if copy_images:
                    shutil.copy2(src_path, dst_path)
                else:
                    os.symlink(os.path.abspath(src_path), dst_path)
            ref_relpaths.append(relpath)

        # Copy/link generated image
        gen_relpath = None
        if gen_image:
            gen_fname = os.path.basename(gen_image)
            gen_dst = os.path.join(ip_img_dir, gen_fname)
            gen_relpath = os.path.join("images", ip_index, gen_fname)
            if not os.path.exists(gen_dst):
                if copy_images:
                    shutil.copy2(gen_image, gen_dst)
                else:
                    os.symlink(os.path.abspath(gen_image), gen_dst)

        # Build training sample
        builder = build_full_training_sample if full_turns else build_simple_training_sample
        sample = builder(parsed, ref_relpaths, gen_relpath)
        samples.append(sample)
        stats["valid"] += 1

    print(f"\n📊 Processing stats:")
    for k, v in sorted(stats.items()):
        print(f"   {k}: {v}")

    if not samples:
        print("❌ No valid samples found. Exiting.")
        return

    # 3. Train/test split
    random.shuffle(samples)
    n_test = max(1, int(len(samples) * test_ratio))
    test_samples = samples[:n_test]
    train_samples = samples[n_test:]

    print(f"\n📝 Split: {len(train_samples)} train / {len(test_samples)} test")

    # 4. Write JSONL
    train_path = os.path.join(output_dir, "train.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    for fpath, data in [(train_path, train_samples), (test_path, test_samples)]:
        with open(fpath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"   ✅ Written {len(data)} samples → {fpath}")

    # 5. Also write a plain-text prompt file (for compatibility with TextPromptDataset)
    # for split_name, data in [("train", train_samples), ("test", test_samples)]:
    #     txt_path = os.path.join(output_dir, f"{split_name}.txt")
    #     with open(txt_path, "w", encoding="utf-8") as f:
    #         for item in data:
    #             f.write(item["image_prompt"].replace("\n", " ") + "\n")
    #     print(f"   ✅ Written {len(data)} prompts → {txt_path}")

    # 6. Summary JSON
    summary = {
        "stage2_dir": stage2_dir,
        "stage3_dir": stage3_dir,
        "total_trajectories": stats["total"],
        "valid_samples": stats["valid"],
        "train_count": len(train_samples),
        "test_count": len(test_samples),
        "stats": dict(stats),
    }
    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Summary → {summary_path}")


# ─────────────────────── CLI ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert raw multi-stage inference outputs to training JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (symlink images)
  python data_process.py \\
    --stage2_dir /apdcephfs_zwfy2/share_303944931/.../output \\
    --stage3_dir /path/to/stage3_generation/final_search_gemini3_v2 \\
    --output_dir ./dataset/interleaved

  # Copy images instead of symlinks, include full dialogue
  python data_process.py \\
    --stage2_dir /apdcephfs_zwfy2/share_303944931/.../output \\
    --stage3_dir /path/to/stage3_generation/final_search_gemini3_v2 \\
    --output_dir ./dataset/interleaved \\
    --copy_images --full_turns

  # Custom test ratio
  python data_process.py \\
    --stage2_dir ./stage2_output \\
    --stage3_dir ./stage3_output \\
    --output_dir ./dataset/interleaved \\
    --test_ratio 0.1
""",
    )
    parser.add_argument("--stage2_dir", required=True,
                        help="Root dir with *_trajectory.json & intermediate/")
    parser.add_argument("--stage3_dir", required=True,
                        help="Dir with generated images ({ip_index}_*.png)")
    parser.add_argument("--output_dir", required=True,
                        help="Output dataset directory")
    parser.add_argument("--test_ratio", type=float, default=0.05,
                        help="Fraction for test split (default: 0.05)")
    parser.add_argument("--copy_images", action="store_true",
                        help="Copy images instead of symlinking")
    parser.add_argument("--full_turns", action="store_true",
                        help="Include assistant responses & tool outputs in turns")
    parser.add_argument("--min_recaption_len", type=int, default=50,
                        help="Skip samples with recaption < N chars (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/test split")

    args = parser.parse_args()

    process_dataset(
        stage2_dir=args.stage2_dir,
        stage3_dir=args.stage3_dir,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        copy_images=args.copy_images,
        full_turns=args.full_turns,
        min_recaption_len=args.min_recaption_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()