from transformers import AutoModel, AutoConfig
import torch
import sys
import torch.nn as nn
import importlib
from pathlib import Path

class VideoMAEv2Classifier(torch.nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.base = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)

        # Use embed_dim from model_config
        hidden_size = config.model_config["embed_dim"]

        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # x = [B, C, T, H, W]
        outputs = self.base(pixel_values=x)

        if isinstance(outputs, torch.Tensor):
            # OpenGVLab style: [B, D] already pooled
            if outputs.ndim == 2:
                pooled = outputs  # [B, D]
            elif outputs.ndim == 3:
                # [B, T, D] → mean over time
                pooled = outputs.mean(1)
            else:
                raise ValueError(f"Unexpected tensor shape: {outputs.shape}")
            return self.classifier(pooled)

        elif hasattr(outputs, "last_hidden_state"):
            pooled = outputs.last_hidden_state.mean(1)  # [B, T, D] → [B, D]
            return self.classifier(pooled)

        elif hasattr(outputs, "logits"):
            return outputs.logits  # Already classified

        else:
            raise TypeError(f"Unexpected model output type: {type(outputs)}")
