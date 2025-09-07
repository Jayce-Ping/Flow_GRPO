from typing import List, Tuple, Optional, Union
import re
import open_clip
import torch
import numpy as np
from PIL import Image
from flow_grpo.rewards.utils import divide_image, extract_grid_info, divide_prompt

class SubfigClipTScorer(torch.nn.Module):
    """
        Scorer for sub-images clip-T-score - align Image-Text semantics.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-H-14",
            pretrained="laion2b_s32b_b79k",
            device=device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")

    @torch.no_grad()
    def __call__(self,
        images: List[Image.Image],
        prompts : List[str],
        metadata : List[dict]
    ) -> np.ndarray:
        """
            Compute the average CLIP score for each subfigure-subprompt pair in a batch of images and prompts.

            Args:
                images (List[Image.Image]): List of PIL images to be evaluated.
                prompts (List[str]): List of prompts corresponding to each image. Each prompt may contain subprompts for subfigures.
                metadata (List[dict]): List of metadata dictionaries for each image (not used in this function).

            Returns:
                np.ndarray: Array of average CLIP scores for each image, computed as the mean of the diagonal of the subfigure-subprompt CLIP score matrix.
        """
        scores = []
        for image, prompt in zip(images, prompts):
            grid_info = extract_grid_info(prompt)
            sub_images = divide_image(image, grid_info)
            sub_prompts = divide_prompt(prompt)[1:]

            clip_matrix = self.compute_ClipT_matrix(sub_prompts, sub_images)
            # clip_scores = clip_matrix.softmax(dim=-1).diagonal().numpy()
            clip_scores = clip_matrix.diagonal().numpy()

            scores.append(np.mean(clip_scores))
        
        return np.array(scores)

    @torch.no_grad()
    def compute_ClipT_matrix(self, text : Union[str, List[str]], image : Union[Image.Image, list[Image.Image]]):
        input_texts = [text] if isinstance(text, str) else text
        input_images = [image] if isinstance(image, Image.Image) else image
        
        text_tokens = self.tokenizer(input_texts).to(self.device)
        images = torch.stack([self.preprocess(img).to(self.device) for img in input_images], dim=0)

        with torch.no_grad():
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            clip_scores = (image_features @ text_features.T)
            return clip_scores.cpu()

def download_model():
    scorer = SubfigClipTScorer(device='cpu')        

def main():
    image_paths = ['/root/flux_trained_0001_0003.png', '/root/flux_0001_0003.png', '/root/sd3_0001_0003.png']
    images = [Image.open(img) for img in image_paths]
    prompt = "THREE-PANEL Images with a 1x3 grid layout a male child with a round face, short ginger hair, and curious, wide eyes, rendered in watercolor style.All illustrations maintain a warm, whimsical watercolor aesthetic with soft edges and vibrant yet gentle colors. The child's features, including ginger hair and wide-eyed curiosity, remain consistent across settings. [LEFT]:The child plays in a sunlit backyard, surrounded by scattered toys and a half-built sandcastle. Dandelion puffs float in the air, and a small dog bounds joyfully nearby. The scene emphasizes playful energy with loose brushstrokes and warm golden-green hues. [MIDDLE]:The child explores a museum exhibit, gazing up at a towering dinosaur skeleton. Display cases glow softly with amber lighting, casting playful shadows. His posture leans forward in wonder, clutching a magnifying glass, with watercolor textures suggesting aged parchment and fossil textures. [RIGHT]:The child sits cross-legged in a wooden treehouse, sketching in a notebook. Sunlight filters through leaves, dappling the pages. A jar of fireflies and binoculars rest beside him, with distant hills rendered in hazy blue layers to evoke depth and quiet imagination."
    prompts = [prompt for _ in range(len(images))]

    scorer = SubfigClipTScorer(device='cuda:0')

    scores = scorer(images, prompts, [])

    print(scores)

if __name__ == "__main__":
    main()