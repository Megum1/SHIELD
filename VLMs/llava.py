import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from .base import VLMBase


class Llava(VLMBase):
    """
    Llava model for visual language tasks.
    Inherits from BaseVLM.
    """

    def __init__(self, model_id: str):
        super().__init__(model_id)
        if self.model_id == "llava-1.5-7b":
            self.load_name = "llava-hf/llava-1.5-7b-hf"
        elif self.model_id == "llava-1.5-13b":
            self.load_name = "llava-hf/llava-1.5-13b-hf"
        elif self.model_id == "llava-1.6-7b":
            self.load_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        elif self.model_id == "llava-1.6-13b":
            self.load_name = "llava-hf/llava-v1.6-vicuna-13b-hf"
        elif self.model_id == "llava-1.6-34b":
            self.load_name = "llava-hf/llava-v1.6-34b-hf"
        else:
            raise ValueError(f"Unsupported model ID: {self.model_id}")

        # Initialize model and processor attributes
        self.load_model()

    def load_model(self):
        if "1.6" in self.model_id:
            load_method = LlavaNextForConditionalGeneration
            processor_method = LlavaNextProcessor
        else:
            load_method = LlavaForConditionalGeneration
            processor_method = AutoProcessor
        self.model = load_method.from_pretrained(
            self.load_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.processor = processor_method.from_pretrained(self.load_name, use_fast=True)

    def process_response(self, response: str) -> str:
        if self.model_id == "llava-1.6-7b":
            sep = "[/INST]"
        elif self.model_id == "llava-1.6-34b":
            sep = "<|im_start|> assistant"
        else:
            sep = "ASSISTANT:"
        # Split the response by the separator and return the last part
        return response.split(sep)[-1].strip()

    def forward(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        # Process the input caption
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image"},],
            },
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        # Prepare inputs
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.model.device, torch.float16
        )

        # Generate output
        if temperature == 0.0:
            kwargs = {"do_sample": False}
        else:
            kwargs = {"do_sample": True, "temperature": temperature}
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
        response = self.processor.decode(output[0], skip_special_tokens=True)
        response = self.process_response(response)

        return response
