"""
Multi-Scale Temporal IvyFake Detector
Analyzes videos at multiple temporal scales for better artifact detection
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class TemporalPyramidExtractor(nn.Module):
    """
    Extracts features at multiple temporal scales
    """
    
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        
        # Three temporal scales
        self.slow_branch = self._make_temporal_branch(embed_dim, name="slow")
        self.medium_branch = self._make_temporal_branch(embed_dim, name="medium")
        self.fast_branch = self._make_temporal_branch(embed_dim, name="fast")
        
        # Cross-scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU()
        )
        
    def _make_temporal_branch(self, embed_dim: int, name: str):
        """Create a temporal processing branch"""
        return nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
    
    def forward(
        self,
        slow_features: torch.Tensor,
        medium_features: torch.Tensor,
        fast_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            slow_features: (batch, num_frames_slow, embed_dim)
            medium_features: (batch, num_frames_medium, embed_dim)
            fast_features: (batch, num_frames_fast, embed_dim)
        
        Returns:
            fused_features: (batch, embed_dim)
        """
        # Process each scale
        # (batch, embed_dim, num_frames) -> (batch, embed_dim, 1) -> (batch, embed_dim)
        slow = self.slow_branch(slow_features.transpose(1, 2)).squeeze(-1)
        medium = self.medium_branch(medium_features.transpose(1, 2)).squeeze(-1)
        fast = self.fast_branch(fast_features.transpose(1, 2)).squeeze(-1)
        
        # Concatenate and fuse
        multi_scale = torch.cat([slow, medium, fast], dim=-1)
        fused = self.scale_fusion(multi_scale)
        
        return fused


class MultiScaleIvyDetector(nn.Module):
    """
    Enhanced IvyFake detector with multi-scale temporal analysis
    """
    
    def __init__(
        self,
        backbone: str = "openai/clip-vit-base-patch32",
        num_classes: int = 2,
        embed_dim: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Vision encoder (shared across scales)
        from transformers import CLIPModel
        clip_model = CLIPModel.from_pretrained(backbone)
        self.vision_encoder = clip_model.vision_model
        
        # Freeze backbone
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # Multi-scale temporal pyramid
        self.temporal_pyramid = TemporalPyramidExtractor(embed_dim)
        
        # Spatial analyzer (for single-frame artifacts)
        self.spatial_analyzer = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + 512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(256, num_classes)
        
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLIP features from images"""
        vision_outputs = self.vision_encoder(pixel_values=images)
        return vision_outputs.pooler_output
    
    def forward(
        self,
        slow_frames: torch.Tensor,
        medium_frames: torch.Tensor,
        fast_frames: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            slow_frames: (batch, num_frames_slow, C, H, W) - Low fps (0.5)
            medium_frames: (batch, num_frames_medium, C, H, W) - Medium fps (1.0)
            fast_frames: (batch, num_frames_fast, C, H, W) - High fps (2.0)
        
        Returns:
            Dict with logits and features
        """
        batch_size = slow_frames.shape[0]
        
        # Extract features at each scale
        def extract_scale_features(frames):
            num_frames = frames.shape[1]
            frames_flat = frames.view(-1, *frames.shape[2:])
            features = self.extract_features(frames_flat)
            return features.view(batch_size, num_frames, -1)
        
        slow_features = extract_scale_features(slow_frames)
        medium_features = extract_scale_features(medium_frames)
        fast_features = extract_scale_features(fast_frames)
        
        # Multi-scale temporal analysis
        temporal_features = self.temporal_pyramid(
            slow_features, medium_features, fast_features
        )
        
        # Spatial analysis (use medium frames as reference)
        spatial_features = self.spatial_analyzer(medium_features.mean(dim=1))
        
        # Fuse temporal and spatial
        combined = torch.cat([temporal_features, spatial_features], dim=-1)
        fused_features = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return {
            'logits': logits,
            'temporal_features': temporal_features,
            'spatial_features': spatial_features
        }
    
    def predict(
        self,
        slow_frames: torch.Tensor,
        medium_frames: torch.Tensor,
        fast_frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions"""
        self.eval()
        with torch.no_grad():
            output = self.forward(slow_frames, medium_frames, fast_frames)
            logits = output['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs


def load_multiscale_detector(
    weights_path: str = None,
    device: str = 'cpu'
) -> MultiScaleIvyDetector:
    """Load pretrained multi-scale detector"""
    model = MultiScaleIvyDetector()
    
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {weights_path}")
    
    model.to(device)
    model.eval()
    return model