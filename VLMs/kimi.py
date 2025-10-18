import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from .base import VLMBase


class KimiVL(VLMBase):
    """
    KimiVL model for visual language tasks.
    Inherits from BaseVLM.
    """

    def __init__(self, model_id: str):
        super().__init__(model_id)
        if self.model_id == "kimi-vl-a3b":
            self.load_name = "moonshotai/Kimi-VL-A3B-Instruct"
        else:
            raise ValueError(f"Unsupported model ID: {self.model_id}")

        # Initialize model and processor attributes
        self.load_model()

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.load_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.load_name, trust_remote_code=True
        )

    def process_response(self, response: str) -> str:
        return response

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
            conversation, add_generation_prompt=True, return_tensors="pt"
        )

        # Prepare inputs
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # Generate output
        do_sample = False if temperature == 0.0 else True
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        response = self.process_response(response)

        return response
