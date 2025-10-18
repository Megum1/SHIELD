import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from .base import VLMBase


class Ovis(VLMBase):
    """
    Ovis model for visual language tasks.
    Inherits from BaseVLM.
    """

    def __init__(self, model_id: str):
        super().__init__(model_id)
        if self.model_id == "ovis2-1b":
            self.load_name = "AIDC-AI/Ovis2-1B"
        elif self.model_id == "ovis2-2b":
            self.load_name = "AIDC-AI/Ovis2-2B"
        elif self.model_id == "ovis2-4b":
            self.load_name = "AIDC-AI/Ovis2-4B"
        elif self.model_id == "ovis2-8b":
            self.load_name = "AIDC-AI/Ovis2-8B"
        elif self.model_id == "ovis2-16b":
            self.load_name = "AIDC-AI/Ovis2-16B"
        elif self.model_id == "ovis2-34b":
            self.load_name = "AIDC-AI/Ovis2-34B"
        else:
            raise ValueError(f"Unsupported model ID: {self.model_id}")

        # Initialize model and processor attributes
        self.load_model()

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.load_name,
            torch_dtype=torch.bfloat16,
            multimodal_max_length=32768,
            # This is required without FlashAttention
            llm_attn_implementation="eager",
            device_map="auto",
            trust_remote_code=True,
        )
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()

    def process_response(self, response: str) -> str:
        return response

    def forward(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        # Single-image input
        images = [image]
        max_partition = 9
        query = f"<image>\n{prompt}"

        # Format conversation
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, images, max_partition=max_partition
        )
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(
                dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device
            )
        pixel_values = [pixel_values]

        # Generate output
        do_sample = False if temperature == 0.0 else True
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True,
            )
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **gen_kwargs,
            )[0]

        # Decode output
        response = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        response = self.process_response(response)
        return response
