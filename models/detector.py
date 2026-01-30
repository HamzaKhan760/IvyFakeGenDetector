"""
IvyFake Detector Model
Unified vision-language model for explainable AIGC detection
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from typing import Dict, Tuple, Optional


class TemporalArtifactAnalyzer(nn.Module):
    """Analyzes temporal inconsistencies in video frames"""
    
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
        
    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_features: (batch, num_frames, embed_dim)
        Returns:
            temporal_features: (batch, embed_dim)
        """
        # Transpose for Conv1d: (batch, embed_dim, num_frames)
        x = frame_features.transpose(1, 2)
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)  # Back to (batch, num_frames, embed_dim)
        
        # Self-attention across frames
        x = x.transpose(0, 1)  # (num_frames, batch, embed_dim)
        attn_out, _ = self.attention(x, x, x)
        attn_out = attn_out.transpose(0, 1)  # (batch, num_frames, embed_dim)
        
        # Global average pooling
        return attn_out.mean(dim=1)


class SpatialArtifactAnalyzer(nn.Module):
    """Analyzes spatial artifacts in individual frames"""
    
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.spatial_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_features: (batch, num_frames, embed_dim)
        Returns:
            spatial_features: (batch, embed_dim)
        """
        # Apply attention and pool
        attention_weights = self.spatial_attention(frame_features)
        attended_features = frame_features * attention_weights
        return attended_features.mean(dim=1)


class IvyXDetector(nn.Module):
    """
    Unified explainable AIGC detector for images and videos
    Based on vision-language architecture with CLIP backbone
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        num_classes: int = 2,
        embed_dim: int = 512,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        # Load CLIP model as vision backbone
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Temporal and spatial artifact analyzers
        self.temporal_analyzer = TemporalArtifactAnalyzer(embed_dim)
        self.spatial_analyzer = SpatialArtifactAnalyzer(embed_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head
        self.classifier = nn.Linear(embed_dim // 2, num_classes)
        
        # Explanation generator (simplified - uses pooled features)
        self.explanation_head = nn.Linear(embed_dim // 2, 768)  # For text generation
        
    def extract_frame_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from frames using CLIP vision encoder"""
        vision_outputs = self.clip_model.vision_model(pixel_values=images)
        return vision_outputs.pooler_output
    
    def forward(
        self,
        images: torch.Tensor,
        return_explanations: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (batch, num_frames, channels, height, width) or (batch, channels, height, width)
            return_explanations: Whether to generate explanation features
        
        Returns:
            Dictionary containing:
                - logits: Classification logits (batch, num_classes)
                - temporal_features: Temporal artifact features (if video)
                - spatial_features: Spatial artifact features
                - explanation_features: Features for explanation generation (optional)
        """
        # Handle both image and video inputs
        if images.dim() == 5:  # Video: (batch, num_frames, C, H, W)
            batch_size, num_frames = images.shape[:2]
            # Flatten batch and frames
            images_flat = images.view(-1, *images.shape[2:])
        else:  # Image: (batch, C, H, W)
            batch_size = images.shape[0]
            num_frames = 1
            images_flat = images
        
        # Extract features
        frame_features = self.extract_frame_features(images_flat)
        
        # Reshape back to (batch, num_frames, embed_dim)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        
        # Analyze temporal and spatial artifacts
        if num_frames > 1:
            temporal_features = self.temporal_analyzer(frame_features)
        else:
            temporal_features = frame_features.squeeze(1)
            
        spatial_features = self.spatial_analyzer(frame_features)
        
        # Fuse features
        combined_features = torch.cat([temporal_features, spatial_features], dim=-1)
        fused_features = self.fusion(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Prepare output
        output = {
            'logits': logits,
            'temporal_features': temporal_features,
            'spatial_features': spatial_features
        }
        
        if return_explanations:
            explanation_features = self.explanation_head(fused_features)
            output['explanation_features'] = explanation_features
        
        return output
    
    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on input images/videos
        
        Returns:
            predictions: Class predictions (0=real, 1=fake)
            probabilities: Confidence scores
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(images)
            logits = output['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs


class SimplifiedIvyDetector(nn.Module):
    """
    Lightweight version for quick inference
    Uses pre-trained vision backbone with simple classifier
    """
    
    def __init__(
        self,
        backbone: str = "openai/clip-vit-base-patch32",
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Load vision encoder
        clip_model = CLIPModel.from_pretrained(backbone)
        self.vision_encoder = clip_model.vision_model
        
        # Freeze backbone
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # Simple classifier
        embed_dim = self.vision_encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch, channels, height, width)
        Returns:
            logits: (batch, num_classes)
        """
        # Extract features
        vision_outputs = self.vision_encoder(pixel_values=images)
        features = vision_outputs.pooler_output
        
        # Classify
        logits = self.classifier(features)
        return logits
    
    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(images)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs


def load_pretrained_ivydetector(
    weights_path: Optional[str] = None,
    device: str = 'cpu'
) -> IvyXDetector:
    """Load pretrained IvyXDetector model"""
    model = IvyXDetector()
    
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {weights_path}")
    
    model.to(device)
    model.eval()
    return model