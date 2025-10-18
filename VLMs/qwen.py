import torch
from PIL import Image

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    from .qwen_vl_utils import process_vision_info

# from qwen_vl_utils import process_vision_info
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)

from .base import VLMBase


class QwenVL(VLMBase):
    """
    QwenVL model for visual language tasks.
    Inherits from BaseVLM.
    """

    def __init__(self, model_id: str):
        super().__init__(model_id)
        if self.model_id == "qwen2.5-vl-3b":
            self.load_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        elif self.model_id == "qwen2.5-vl-7b":
            self.load_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        elif self.model_id == "qwen2.5-vl-32b":
            self.load_name = "Qwen/Qwen2.5-VL-32B-Instruct"
        else:
            raise ValueError(f"Unsupported model ID: {self.model_id}")

        # Initialize model and processor attributes
        self.load_model()

    def load_model(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.load_name,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.load_name)

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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image,},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference
        if temperature == 0.0:
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        else:
            # Use temperature for sampling
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
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
