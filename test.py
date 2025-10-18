import os
import time
import json
import argparse
from PIL import Image
from typing import List

from utils import seed_torch, VLM_DICT
from prompts import prompt_direct, prompt_cot

# Filter out warnings for cleaner output
import warnings

warnings.filterwarnings("ignore")


class VLMDetector:
    def __init__(self, model_id, sampling_test_times=5, sampling_temperature=0.7):
        if model_id.startswith("llava"):
            from VLMs.llava import Llava

            assert (
                model_id in VLM_DICT["llava"]
            ), f"Model ID {model_id} is not supported by Llava."
            self.vlm = Llava(model_id)
        elif model_id.startswith("llama"):
            from VLMs.llama import Llama

            assert (
                model_id in VLM_DICT["llama"]
            ), f"Model ID {model_id} is not supported by Llama."
            self.vlm = Llama(model_id)
        elif model_id.startswith("deepseek"):
            from VLMs.deepseek import DeepSeek

            assert (
                model_id in VLM_DICT["deepseek"]
            ), f"Model ID {model_id} is not supported by DeepSeek."
            self.vlm = DeepSeek(model_id)
        elif model_id.startswith("intern"):
            from VLMs.intern import InternVL

            assert (
                model_id in VLM_DICT["intern"]
            ), f"Model ID {model_id} is not supported by InternVL."
            self.vlm = InternVL(model_id)
        elif model_id.startswith("qwen"):
            from VLMs.qwen import QwenVL

            assert (
                model_id in VLM_DICT["qwen"]
            ), f"Model ID {model_id} is not supported by QwenVL."
            self.vlm = QwenVL(model_id)
        elif model_id.startswith("kimi"):
            from VLMs.kimi import KimiVL

            assert (
                model_id in VLM_DICT["kimi"]
            ), f"Model ID {model_id} is not supported by KimiVL."
            self.vlm = KimiVL(model_id)
        elif model_id.startswith("ovis"):
            from VLMs.ovis import Ovis

            assert (
                model_id in VLM_DICT["ovis"]
            ), f"Model ID {model_id} is not supported by Ovis."
            self.vlm = Ovis(model_id)
        else:
            raise ValueError(f"Model ID {model_id} is not recognized.")

        # Configure the model for evaluation
        self.sampling_test_times = sampling_test_times
        self.sampling_temperature = sampling_temperature

    def forward(
        self,
        image_filepath: str,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """
        Forward pass through the VLM for a given image and prompt.
        
        Args:
            image_filepath: Path to the input image file.
            prompt: A text prompt to guide the model.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Temperature for generation (0.0 for greedy decoding).
        
        Returns:
            The output from the model for the image.
        """
        # Load the image
        # Can customize the image loading process if needed
        image = Image.open(image_filepath).convert("RGB")
        # Forward pass through the model
        return self.vlm.forward(image, prompt, max_new_tokens, temperature)

    def evaluate(
        self,
        image_filepath: str,
        phase: str = "direct",
        mode: str = "greedy",
        max_new_tokens: int = 256,
    ) -> List[str]:
        """
        Evaluate the model on the image file list.
        
        Args:
            image_filepath: Path to the input image file.
            phase: The phase of evaluation: 'direct' or 'cot'.
            mode: The mode of evaluation: 'greedy' or 'sampling'.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            The predictions from the model for the image.
        """
        # Select the proper user prompt
        if phase == "direct":
            prompt = prompt_direct
        elif phase == "cot":
            prompt = prompt_cot
        else:
            raise ValueError(f"Phase {phase} is not supported. Use 'direct' or 'cot'.")

        # Evaluate the model on the image file list
        if mode == "greedy":
            output = self.forward(
                image_filepath, prompt, max_new_tokens=max_new_tokens, temperature=0.0
            )
            return [output]
        elif mode == "sampling":
            outputs = []
            for _ in range(self.sampling_test_times):
                output = self.forward(
                    image_filepath,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=self.sampling_temperature,
                )
                outputs.append(output)
            return outputs
        else:
            raise ValueError(
                f"Mode {mode} is not supported. Use 'greedy' or 'sampling'."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Detector Test Script")
    parser.add_argument(
        "--model_id", type=str, required=True, help="ID of the VLM model to use."
    )
    parser.add_argument(
        "--image_filepath",
        type=str,
        default="data/example.png",
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Directory to save the results.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="direct",
        choices=["direct", "cot"],
        help="Phase of evaluation: 'direct' or 'cot'.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="greedy",
        choices=["greedy", "sampling"],
        help="Mode of evaluation: 'greedy' or 'sampling'.",
    )
    parser.add_argument(
        "--max_new_tokens_direct",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate for direct prompting.",
    )
    parser.add_argument(
        "--max_new_tokens_cot",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate for CoT prompting.",
    )
    parser.add_argument(
        "--sampling_test_times",
        type=int,
        default=5,
        help="Number of times to run for sampling inference.",
    )
    parser.add_argument(
        "--sampling_temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling inference.",
    )
    parser.add_argument(
        "--seed", type=int, default=1024, help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    # Set the random seed for reproducibility
    seed_torch(args.seed)

    # Initialize the VLM detector
    model = VLMDetector(args.model_id)

    # Select the maximum number of new tokens based on the phase
    if args.phase == "direct":
        max_new_tokens = args.max_new_tokens_direct
    elif args.phase == "cot":
        max_new_tokens = args.max_new_tokens_cot
    else:
        raise ValueError(
            f"Phase {args.phase} is not supported. Use 'direct' or 'cot' prompting."
        )

    # Begin evaluation
    print(
        f"Starting evaluation with model {args.model_id} in phase '{args.phase}' using mode '{args.mode}'..."
    )
    start_time = time.time()

    # Evaluate the model
    predictions = model.evaluate(
        image_filepath=args.image_filepath,
        phase=args.phase,
        mode=args.mode,
        max_new_tokens=max_new_tokens,
    )

    # Dump the predictions
    save_dirpath = os.path.join(args.result_dir, args.model_id, args.phase, args.mode)
    os.makedirs(save_dirpath, exist_ok=True)
    with open(
        os.path.join(
            save_dirpath, f"{os.path.basename(args.image_filepath).split('.')[0]}.json"
        ),
        "w",
    ) as f:
        json.dump(predictions, f, indent=4)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
