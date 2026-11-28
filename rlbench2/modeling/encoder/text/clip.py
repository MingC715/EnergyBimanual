from pathlib import Path

import torch
from torch import nn
import transformers


class ClipTokenizer:

    def __init__(self):
        super().__init__()
        model_id = Path(__file__).resolve().parents[3] / "assets" / "clip-vit-base-patch32"
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(model_id)

    @torch.inference_mode()
    def __call__(self, instructions):
        # Handle empty or None instructions
        if instructions is None or (isinstance(instructions, list) and len(instructions) == 0):
            instructions = [""]
        elif isinstance(instructions, str) and instructions == "":
            instructions = [""]
        elif isinstance(instructions, list) and any(inst is None or inst == "" for inst in instructions):
            # Replace None or empty strings with empty string
            instructions = [inst if inst is not None and inst != "" else "" for inst in instructions]

        tokens = self.tokenizer(
            instructions,
            padding="longest",
            return_tensors="pt"
        )["input_ids"]

        return tokens


class ClipTextEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        model_id = Path(__file__).resolve().parents[3] / "assets" / "clip-vit-base-patch32"
        self.model = transformers.CLIPTextModel(transformers.CLIPTextConfig.from_pretrained(model_id))
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(model_id)

    def forward(self, text_or_tokens):
        """
        Forward pass that accepts either text strings or pre-tokenized tokens.
        
        Args:
            text_or_tokens: Either a list of strings or a tensor of token IDs
        """
        # Check if input is text (list of strings) or tokens (tensor)
        if isinstance(text_or_tokens, (list, str)):
            # Tokenize text input
            if isinstance(text_or_tokens, str):
                text_or_tokens = [text_or_tokens]
            tokens = self.tokenizer(
                text_or_tokens,
                padding="longest",
                return_tensors="pt"
            )["input_ids"]
            # Move tokens to same device as model
            tokens = tokens.to(self.model.device)
        else:
            # Already tokenized
            tokens = text_or_tokens

        return self.model(input_ids=tokens).last_hidden_state
