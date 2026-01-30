"""
Preprocessing utilities for IvyFake detector
Handles video/image loading, face detection, and frame extraction
"""

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Union
from facenet_pytorch import MTCNN


class VideoPreprocessor:
    """Handles video preprocessing for IvyFake detector"""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        max_frames: int = 16,
        use_face_detection: bool = True,
        device: str = 'cpu'
    ):
        self.target_size = target_size
        self.max_frames = max_frames
        self.use_face_detection = use_face_detection
        self.device = device
        
        if use_face_detection:
            try:
                self.face_detector = MTCNN(
                    keep_all=False,
                    device=device,
                    post_process=False
                )
            except Exception as e:
                print(f"Warning: Could not initialize MTCNN: {e}")
                print("Falling back to full-frame processing")
                self.use_face_detection = False
                self.face_detector = None
        else:
            self.face_detector = None
    
    def extract_frames(
        self,
        video_path: Union[str, Path],
        uniform_sampling: bool = True
    ) -> List[np.ndarray]:
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            uniform_sampling: If True, sample frames uniformly; else use first N frames
        
        Returns:
            List of frames as numpy arrays (H, W, C)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if uniform_sampling:
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
        else:
            # Take first N frames
            frame_indices = list(range(min(self.max_frames, total_frames)))
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")
        
        return frames
    
    def detect_and_crop_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face in frame and crop
        
        Args:
            frame: Input frame (H, W, C)
        
        Returns:
            Cropped face region or None if no face detected
        """
        if not self.use_face_detection or self.face_detector is None:
            return frame
        
        try:
            # Convert to PIL Image
            pil_frame = Image.fromarray(frame)
            
            # Detect face
            boxes, _ = self.face_detector.detect(pil_frame)
            
            if boxes is not None and len(boxes) > 0:
                # Take the first (most prominent) face
                box = boxes[0]
                x1, y1, x2, y2 = map(int, box)
                
                # Add margin
                margin = 20
                h, w = frame.shape[:2]
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                
                # Crop face
                face = frame[y1:y2, x1:x2]
                return face
            else:
                # No face detected, return full frame
                return frame
                
        except Exception as e:
            print(f"Face detection error: {e}")
            return frame
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess single frame: resize and normalize
        
        Args:
            frame: Input frame (H, W, C) in RGB
        
        Returns:
            Preprocessed frame (C, H, W) normalized to [0, 1]
        """
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
    ) -> torch.Tensor:
        """
        Complete video preprocessing pipeline
        
        Args:
            video_path: Path to video file
        
        Returns:
            Preprocessed video tensor (1, num_frames, C, H, W)
        """
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Process each frame
        processed_frames = []
        for frame in frames:
            # Detect and crop face
            if self.use_face_detection:
                frame = self.detect_and_crop_face(frame)
            
            # Preprocess
            if frame is not None:
                processed_frame = self.preprocess_frame(frame)
                processed_frames.append(processed_frame)
        
        # Stack frames
        video_tensor = np.stack(processed_frames, axis=0)  # (num_frames, C, H, W)
        video_tensor = torch.from_numpy(video_tensor).unsqueeze(0)  # (1, num_frames, C, H, W)
        
        return video_tensor
    
    def process_image(
        self,
        image_path: Union[str, Path]
    ) -> torch.Tensor:
        """
        Process single image
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed image tensor (1, C, H, W)
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect and crop face
        if self.use_face_detection:
            image = self.detect_and_crop_face(image)
        
        # Preprocess
        if image is not None:
            processed_image = self.preprocess_frame(image)
            image_tensor = torch.from_numpy(processed_image).unsqueeze(0)  # (1, C, H, W)
            return image_tensor
        else:
            raise ValueError("Failed to process image")


class DataAugmentation:
    """Data augmentation for training"""
    
    @staticmethod
    def apply_compression(frame: np.ndarray, quality: int = 75) -> np.ndarray:
        """Apply JPEG compression"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return decoded
    
    @staticmethod
    def apply_blur(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def apply_noise(frame: np.ndarray, noise_level: float = 10) -> np.ndarray:
        """Add Gaussian noise"""
        noise = np.random.randn(*frame.shape) * noise_level
        noisy_frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        return noisy_frame
    
    @staticmethod
    def random_augment(frame: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Randomly apply augmentations"""
        if np.random.rand() < p:
            aug_type = np.random.choice(['compress', 'blur', 'noise'])
            if aug_type == 'compress':
                quality = np.random.randint(50, 95)
                frame = DataAugmentation.apply_compression(frame, quality)
            elif aug_type == 'blur':
                kernel_size = np.random.choice([3, 5, 7])
                frame = DataAugmentation.apply_blur(frame, kernel_size)
            else:
                noise_level = np.random.uniform(5, 15)
                frame = DataAugmentation.apply_noise(frame, noise_level)
        return frame


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor for CLIP model
    
    Args:
        tensor: Input tensor (B, C, H, W) or (B, T, C, H, W)
    
    Returns:
        Normalized tensor
    """
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, -1, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, -1, 1, 1)
    
    if tensor.dim() == 5:  # Video
        mean = mean.unsqueeze(1)
        std = std.unsqueeze(1)
    
    return (tensor - mean) / std