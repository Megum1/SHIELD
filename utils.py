import os
import torch
import random
import numpy as np


# Set random seed
def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


VLM_DICT = {
    # Llava series
    "llava": [
        "llava-1.5-7b",
        "llava-1.5-13b",
        "llava-1.6-7b",
        "llava-1.6-13b",
        "llava-1.6-34b",
    ],
    # Llama3.2 series
    "llama": ["llama-3.2-11b",],
    # DeepSeekVL series
    "deepseek": ["deepseek-vl2-tiny", "deepseek-vl2-small", "deepseek-vl2",],
    # InternVL3 series
    "intern": [
        "internvl-3-1b",
        "internvl-3-2b",
        "internvl-3-8b",
        "internvl-3-14b",
        "internvl-3-38b",
    ],
    # QwenVL2.5 series
    "qwen": ["qwen2.5-vl-3b", "qwen2.5-vl-7b", "qwen2.5-vl-32b",],
    # KimiVL series
    "kimi": ["kimi-vl-a3b",],
    # Ovis2 series
    "ovis": ["ovis2-1b", "ovis2-2b", "ovis2-4b", "ovis2-8b", "ovis2-16b", "ovis2-34b",],
}
