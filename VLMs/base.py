# This is the base class for all VLMs (Vision Language Models).

from PIL import Image


class VLMBase:
    def __init__(self, model_id: str):
        """
        Initialize the VLM with the given model ID.
        
        Args:
            model_id: The ID of the model to be used.
        """
        self.model_id = model_id

    def load_model(self):
        """
        Load the VLM model.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def process_response(self, response: str) -> str:
        """
        Process the response from the model.
        
        Args:
            response: The raw response string from the model.
        
        Returns:
            A parsed or formatted response string without template tokens.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def forward(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """
        Forward pass through the model for a batch of images and prompts.
        
        Args:
            image: An input image to the model.
            prompt: A text prompt to guide the model.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Temperature for generation (0.0 for greedy decoding).
        
        Returns:
            The output from the model for the image and prompt pair.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
