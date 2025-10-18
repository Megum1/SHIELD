import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

from .base import VLMBase


class Llama(VLMBase):
    """
    Llama model for visual language tasks.
    Inherits from BaseVLM.
    """

    def __init__(self, model_id: str):
        super().__init__(model_id)
        if self.model_id == "llama-3.2-11b":
            self.load_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        else:
            raise ValueError(f"Unsupported model ID: {self.model_id}")

        # Initialize model and processor attributes
        self.load_model()

    def load_model(self):
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.load_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.load_name)

    def process_response(self, response: str) -> str:
        find_id = "assistant"
        # Find the first occurrence of find_id in the response
        index = response.find(find_id)
        if index != -1:
            # Return the substring starting from the index of find_id
            return response[index + len(find_id) :].strip()
        return response.strip()

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
                "content": [{"type": "image"}, {"type": "text", "text": prompt},],
            },
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        # Prepare inputs
        inputs = self.processor(
            image, prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)

        # Generate output
        do_sample = False if temperature == 0.0 else True
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        response = self.processor.decode(output[0], skip_special_tokens=True)
        response = self.process_response(response)

        return response
