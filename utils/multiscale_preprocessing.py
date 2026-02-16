"""
Multi-scale preprocessing for temporal pyramid analysis
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Union
from .preprocessing import VideoPreprocessor


class MultiScaleVideoPreprocessor:
    """
    Extracts frames at multiple temporal scales (fps rates)
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        slow_fps: float = 0.5,    # Long-term consistency
        medium_fps: float = 1.0,   # Standard rate
        fast_fps: float = 2.0,     # Micro-movements
        max_duration: int = 10,    # Max video duration (seconds)
        use_face_detection: bool = False,
        device: str = 'cpu'
    ):
        self.target_size = target_size
        self.slow_fps = slow_fps
        self.medium_fps = medium_fps
        self.fast_fps = fast_fps
        self.max_duration = max_duration
        self.use_face_detection = use_face_detection
        self.device = device
        
        # Base preprocessor for face detection
        self.base_preprocessor = VideoPreprocessor(
            target_size=target_size,
            max_frames=int(max_duration * medium_fps),
            use_face_detection=use_face_detection,
            device=device
        )
    
    def extract_frames_at_fps(
        self,
        video_path: Union[str, Path],
        fps: float
    ) -> List[np.ndarray]:
        """
        Extract frames at specific fps rate
        
        Args:
            video_path: Path to video
            fps: Frames per second to extract
        
        Returns:
            List of frames
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        # Calculate frame indices
        actual_duration = min(duration, self.max_duration)
        num_frames = int(actual_duration * fps)
        frame_interval = video_fps / fps
        
        frames = []
        for i in range(num_frames):
            frame_idx = int(i * frame_interval)
            
            if frame_idx >= total_frames:
                break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")
        
        return frames
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess single frame"""
        # Detect and crop face if enabled
        if self.use_face_detection:
            frame = self.base_preprocessor.detect_and_crop_face(frame)
        
        if frame is None:
            # Return blank frame if processing failed
            return np.zeros((3, *self.target_size), dtype=np.float32)
        
        # Resize
        frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Transpose to (C, H, W)
        frame = np.transpose(frame, (2, 0, 1))
        
        return frame
    
    def process_video(
        self,
        video_path: Union[str, Path]
    ) -> Dict[str, torch.Tensor]:
        """
        Process video at multiple temporal scales
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with tensors at different scales:
                - 'slow': (1, num_frames_slow, C, H, W)
                - 'medium': (1, num_frames_medium, C, H, W)
                - 'fast': (1, num_frames_fast, C, H, W)
        """
        # Extract frames at each scale
        slow_frames = self.extract_frames_at_fps(video_path, self.slow_fps)
        medium_frames = self.extract_frames_at_fps(video_path, self.medium_fps)
        fast_frames = self.extract_frames_at_fps(video_path, self.fast_fps)
        
        # Preprocess each scale
        def process_scale(frames):
            processed = [self.preprocess_frame(f) for f in frames]
            tensor = np.stack(processed, axis=0)  # (num_frames, C, H, W)
            return torch.from_numpy(tensor).unsqueeze(0)  # (1, num_frames, C, H, W)
        
        return {
            'slow': process_scale(slow_frames),
            'medium': process_scale(medium_frames),
            'fast': process_scale(fast_frames)
        }
    
    def process_image(
        self,
        image_path: Union[str, Path]
    ) -> Dict[str, torch.Tensor]:
        """
        Process single image (duplicated across scales for compatibility)
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary with image tensor at all scales
        """
        # Load and process image
        image_tensor = self.base_preprocessor.process_image(image_path)
        
        # Duplicate for each scale (images don't have temporal dimension)
        return {
            'slow': image_tensor.unsqueeze(1),      # (1, 1, C, H, W)
            'medium': image_tensor.unsqueeze(1),
            'fast': image_tensor.unsqueeze(1)
        }