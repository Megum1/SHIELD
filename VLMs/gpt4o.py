import base64
import requests

from .base import VLMBase


class GPT4o(VLMBase):
    """
    GPT-4o model for visual language tasks.
    Inherits from BaseVLM.
    """

    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.APIKey = "<YOUR_OPENAI_API_KEY>"  # Replace with your actual API key

    # Function to encode the image
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def forward(
        self,
        image_filepath: str,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        # Getting the base64 string
        base64_image = self.encode_image(image_filepath)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.APIKey}",
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )

        try:
            return response.json()["choices"][0]["message"]["content"]
        except:
            return "Error"
