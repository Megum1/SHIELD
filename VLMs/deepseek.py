import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from .deepseek_vl2.models import DeepseekVLV2Processor

from .base import VLMBase


class DeepSeek(VLMBase):
    """
    DeepSeek model for visual language tasks.
    Inherits from BaseVLM.
    """

    def __init__(self, model_id: str):
        super().__init__(model_id)
        if self.model_id == "deepseek-vl2-tiny":
            self.load_name = "deepseek-ai/deepseek-vl2-tiny"
        elif self.model_id == "deepseek-vl2-small":
            self.load_name = "deepseek-ai/deepseek-vl2-small"
        elif self.model_id == "deepseek-vl2":
            self.load_name = "deepseek-ai/deepseek-vl2"
        else:
            raise ValueError(f"Unsupported model ID: {self.model_id}")

        # Initialize model and processor attributes
        self.device_map = self.split_model()
        self.load_model()

    # Split more for deepseek-vl2
    def split_model(self):
        device_map = {}
        num_layers_per_gpu = [15, 15]
        num_layers = sum(num_layers_per_gpu)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f"language.model.layers.{layer_cnt}"] = i
                layer_cnt += 1
        device_map["vision"] = 0
        device_map["projector"] = 0
        device_map["image_newline"] = 0
        device_map["view_seperator"] = 0
        device_map["language.model.embed_tokens"] = 0
        device_map["language.model.norm"] = 0
        device_map["language.lm_head"] = 0
        device_map[f"language.model.layers.{num_layers - 1}"] = 0
        return device_map

    def load_model(self):
        self.vl_chat_processor = DeepseekVLV2Processor.from_pretrained(self.load_name)
        self.vl_gpt = (
            AutoModelForCausalLM.from_pretrained(
                self.load_name, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
            .cuda()
            .eval()
        )

    def process_response(self, response: str) -> str:
        # No special processing needed for DeepSeek
        return response

    def forward(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        conversation = [
            {"role": "<|User|>", "content": f"<image>\n {prompt}", "images": "images",},
            {"role": "<|Assistant|>", "content": ""},
        ]

        # Load images and prepare for inputs
        pil_images = [image]
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt="",
        ).to(self.vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        do_sample = False if temperature == 0.0 else True
        # generate the response
        tokenizer = self.vl_chat_processor.tokenizer
        outputs = self.vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            use_cache=True,
        )

        response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        response = self.process_response(response)
        return response
