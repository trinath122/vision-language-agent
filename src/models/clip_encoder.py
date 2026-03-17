import torch
import torch.nn as nn
import open_clip
from typing import Optional


class CLIPVisionEncoder(nn.Module):
    """
    Vision encoder using CLIP ViT-L/14.
    Encodes images into embeddings aligned with the LLM projection space.
    """

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai"):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.embed_dim = self.model.visual.output_dim

        # Freeze CLIP weights — only train projection layer
        for param in self.model.parameters():
            param.requires_grad = False

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Returns normalized image embeddings. Shape: (B, embed_dim)"""
        with torch.no_grad():
            features = self.model.encode_image(images)
        return features / features.norm(dim=-1, keepdim=True)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Returns normalized text embeddings for contrastive alignment."""
        tokens = self.tokenizer(texts)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image(images)


class VisionProjection(nn.Module):
    """
    Projects CLIP image embeddings into LLaMA's token embedding space.
    Used to bridge vision encoder → language model.
    """

    def __init__(self, clip_dim: int, llm_dim: int, num_patches: int = 1):
        super().__init__()
        self.num_patches = num_patches
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, llm_dim * 2),
            nn.GELU(),
            nn.Linear(llm_dim * 2, llm_dim * num_patches),
        )

    def forward(self, clip_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_embeds: (B, clip_dim)
        Returns:
            projected: (B, num_patches, llm_dim) — visual tokens for LLM
        """
        projected = self.projection(clip_embeds)
        return projected.view(projected.size(0), self.num_patches, -1)
