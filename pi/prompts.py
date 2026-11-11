"""Prompts shared by run.py and hf_reference.py."""
import os
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Photosynthesis is the process by which",
    "<|im_start|>user\nName three primary colors.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
]
NEW_TOKENS = 24
# Public text for calibrating the merged layers (run.py merge). Not one of the test prompts.
CALIB = ("The river runs through the old town before it reaches the sea. Every spring the water rises, and the people "
         "who live along the banks move their boats to higher ground. In 1987 a new bridge was built to replace the "
         "wooden one. It has two lanes for cars and a path for people who walk or ride a bicycle.")

def encode(path, text):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(path, "tokenizer.json")).encode(text).ids

def decode(path, ids):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(path, "tokenizer.json")).decode(ids)
