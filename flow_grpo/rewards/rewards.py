from collections import defaultdict
from typing import List, Tuple, Callable, Union, Dict, Optional
import io
import inspect

from PIL import Image
import numpy as np
import torch
from openai import OpenAI, AsyncOpenAI
from flow_grpo.utils import tensor_list_to_pil_image, tensor_to_pil_image, numpy_to_pil_image, numpy_list_to_pil_image

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):

        buffers = [io.BytesIO() for _ in images]

        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)

        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn

def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta

    return _fn

def aesthetic_score():
    from flow_grpo.rewards.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        scores = scorer(images)
        return scores, {}

    return _fn

def clip_score(device):
    from flow_grpo.rewards.clip_scorer import ClipScorer

    scorer = ClipScorer(dtype=torch.float32).to(device)

    def _fn(images: List[Image.Image], prompts: List[str], metadata: List[dict]) -> Tuple[np.ndarray, dict]:
        # Convert PIL images to pixel tensors in [0, 1] range
        images = np.stack([np.array(img) / 255.0 for img in images])
        images = torch.tensor(images, dtype=torch.float32).to(device)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def image_similarity_score(device):
    from flow_grpo.rewards.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device).cuda()

    def _fn(images, ref_images):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8)/255.0
        scores = scorer.image_similarity(images, ref_images)
        return scores, {}

    return _fn

def pickscore_score(device):
    from flow_grpo.rewards.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images : List[Image.Image], prompts : List[str], metadata : List[dict]) -> Tuple[List[float], dict]:
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def imagereward_score(device):
    from flow_grpo.rewards.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):

        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def grid_layout_score():
    import asyncio
    from flow_grpo.rewards.layout_scorer import GridLayoutScorer

    client = AsyncOpenAI(
        api_key='dummy-key',
        base_url='http://127.0.0.1:8000/v1'
    )

    def _fn(images : List[Image.Image], prompts : List[str], metadatas : List[dict]) -> Tuple[List[float], dict]:
        # Create the GridLayoutScorer instance inside the function, to create its own semaphore for this call
        scorer = GridLayoutScorer(
            client=client,
            model='Qwen2.5-VL-7B-Instruct',
            max_concurrent=300, # Adjust based on the system's capabilities (especially when using vllm as local model server)
        )
        scores = scorer(images, prompts, metadatas)
        return scores, {}

    return _fn

def consistency_score():
    import asyncio
    from flow_grpo.rewards.consistency_scorer import ConsistencyScorer

    client = AsyncOpenAI(
        api_key='dummy-key',
        base_url='http://127.0.0.1:8000/v1'
    )

    def _fn(images : List[Image.Image], prompts : List[str], metadatas : List[dict]) -> Tuple[List[float], dict]:
        # Create the ConsistencyScorer instance inside the function, to create its own semaphore for this call
        scorer = ConsistencyScorer(
            client=client,
            model='Qwen2.5-VL-7B-Instruct',
            max_concurrent=300, # Adjust based on the system's capabilities (especially when using vllm as local model server)
        )
        scores = scorer(images, prompts, metadatas)
        return scores, {}

    return _fn

def subfig_clipT_score(device):
    from flow_grpo.rewards.subfig_clipT import SubfigClipTScorer

    scorer = SubfigClipTScorer(device=device)

    def _fn(images : List[Image.Image], prompts : List[str], metadatas : List[dict]) -> Tuple[np.ndarray, dict]:
        scores = scorer(images, prompts, metadatas)
        return scores, {}

    return _fn

def ocr_score(device):
    from flow_grpo.rewards.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images : List[Image.Image], prompts : List[str], metadata : List[dict]):
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def multi_score(
    device: str,
    score_dict: Dict[str, float],
    aggregate_fn: Optional[Callable[[Dict[str, float]], float]] = None,
) -> Callable[[List[Image.Image], List[str], List[dict]], Tuple[dict[str, np.ndarray], dict]]:
    """
    Constructs a multi-score reward function that computes multiple reward metrics for a batch of images and prompts.

    Args:
        device: The device (e.g., "cuda" or "cpu") on which to run the reward functions.
        
        score_dict (List[str]): A dictionary mapping reward function names to their weights.
        
        aggregate_fn (Callable[[Dict[str, float]], float], optional): A function to aggregate multiple scores.
            If None, defaults to summing all values. The function should accept keyword arguments where:
            - Each keyword corresponds to a key in score_dict (e.g., "clipscore", "aesthetic").
            - Each value is the weighted score (original_score * weight) for that reward metric.
            - Returns a single float representing the final aggregated score.

            Examples:
            - lambda **kwargs: np.sum(list(kwargs.values()))  # Sum all weighted scores (default)
            - lambda **kwargs: np.mean(list(kwargs.values()))  # Average of weighted scores  
            - lambda clipscore, aesthetic: np.exp(clipscore) + np.exp(aesthetic)  # Custom weighting
            - lambda **kwargs: max(kwargs.values())  # Take maximum score

    Returns:
        Callable: A function that takes as input:
            - images (List[Image.Image] or np.ndarray or torch.Tensor): The batch of images to evaluate.
            - prompts (List[str]): The corresponding text prompts for the images.
            - metadata (List[dict]): Additional metadata for each image/prompt pair.
            - ref_images (optional): Reference images for similarity-based rewards.

        The returned function outputs:
            - A dictionary mapping reward names to their computed numpy arrays, including an "avg" key for the aggregated score.
            - A dictionary containing detailed reward information (e.g., per-group or strict scores).

    Raises:
        ValueError: If an unknown score name is provided in score_dict.

    Example:
        reward_fn = multi_score("cuda:0", {"clipscore": 0.5, "aesthetic": 0.5}, aggregate_fn=lambda score1, score2: score1 + score2)
        rewards, details = reward_fn(images, prompts, metadata)
    """
    if aggregate_fn is None:
        # If not given, use np.sum directly
        aggregate_fn = lambda **kwargs: np.sum(list(kwargs.values()))

    assert aggregate_fn is not None

    score_functions = {
        "ocr": ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "clipscore": clip_score,
        "consistency_score": consistency_score,
        "subfig_clipT": subfig_clipT_score,
        "grid_layout": grid_layout_score,
    }

    score_fns = {}

    for score_name, weight in score_dict.items():
        factory = score_functions.get(score_name)
        if factory is None:
            raise ValueError(f"Unknown score: {score_name}")
        params = inspect.signature(factory).parameters
        if "device" in params:
            score_fns[score_name] = factory(device)
        else:
            score_fns[score_name] = factory()

    def _fn(
        images : Union[List[Image.Image], torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray]],
        prompts : List[str],
        metadata: List[dict]
    ) -> Tuple[dict[str, np.ndarray], dict]:
        total_scores = []
        score_details = {}

        # Convert images to PIL format if they are tensors or numpy arrays
        if isinstance(images, torch.Tensor):
            images = tensor_to_pil_image(images)
        elif isinstance(images, np.ndarray):
            images = numpy_to_pil_image(images)
        elif isinstance(images, list) and all(isinstance(img, torch.Tensor) for img in images):
            images = tensor_list_to_pil_image(images)
        elif isinstance(images, list) and all(isinstance(img, np.ndarray) for img in images):
            images = numpy_list_to_pil_image(images)

        assert all(isinstance(img, Image.Image) for img in images), "All images must be a list of PIL Image, or a numpy array / torch Tensor, or a list of them."

        for score_name, weight in score_dict.items():
            scores, rewards = score_fns[score_name](images, prompts, metadata)

            # Make sure to convert all scores to numpy arrays
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            
            if isinstance(scores, list):
                scores = np.array(scores)

            score_details[score_name] = scores
            # Scale each reward by corresponding weight
            total_scores.append(weight * scores)

        # Aggregate scores from different reward models
        total_scores = np.array([
            aggregate_fn(**{k: v for k,v in zip(score_dict.keys(), scores)})
            for scores in zip(*total_scores)
        ])

        score_details['avg'] = total_scores
        return score_details, {}

    return _fn

def main():
    import torchvision.transforms as transforms

    image_paths = [
        "nasa.jpg",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = [{}]  # Example metadata
    score_dict = {
        "clipscore": 1.0
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()