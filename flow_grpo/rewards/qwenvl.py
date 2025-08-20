import torch
import asyncio
import base64
import re
from typing import Union
from io import BytesIO
from PIL import Image
from openai import AsyncOpenAI, OpenAI
import logging

# VLLM log filter
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


def pil_image_to_base64(image, format="JPEG"):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image/{format.lower()};base64,{encoded_image_text}"
    return base64_qwen

def extract_scores(output_texts : Union[list[str], str]):
    if isinstance(output_texts, str):
        output_texts = [output_texts]
        return_list = False
    else:
        return_list = True

    scores = []
    for text in output_texts:
        match = re.search(r'<Score>(\d+)</Score>', text)
        if match:
            scores.append(float(match.group(1))/5)
        else:
            scores.append(0)

    return scores if return_list else scores[0]

class QwenVLScorer:
    def __init__(self, api_key='dummy_key', base_url='http://127.0.0.1:8000/v1', model_name='QwenVL2.5-7B-Instruct'):
        self.openai_api_key = api_key
        self.openai_base_url = base_url
        self.model_name = model_name
        self.query = "Please rate how well the image matches the text on a scale of 1 to 5, with 5 being the best. Only respond with the score in the format <Score>n</Score>, where n is the score."

        self.client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_base_url
        )


    @torch.no_grad()
    def __call__(self, prompts: list[str], images: list[Image.Image]):
        images_base64 = [pil_image_to_base64(image) for image in images]
        responses = []
        for prompt, base64_qwen in zip(prompts, images_base64):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": base64_qwen}},
                        {"type": "text", "text": prompt + "\n" + self.query},
                    ],
                },
            ]
            resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=32,
                    temperature=0.0,
            )
            responses.append(resp)
        
        output_texts = [resp.choices[0].message.content for resp in responses]
        return extract_scores(output_texts)