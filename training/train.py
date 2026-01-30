"""
Training script for IvyFake detector
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from typing import Dict, List, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.detector import SimplifiedIvyDetector
from utils.preprocessing import VideoPreprocessor, normalize_tensor


class VideoDataset(Dataset):
    """Dataset for video/image classification"""
    
    def __init__(
        self,
        data_dir: Path,
        preprocessor: VideoPreprocessor,
        is_video: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.is_video = is_video
        
        # Load file paths and labels
        self.samples = []
        
        # Real samples (label 0)
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for file_path in real_dir.glob('*'):
                if file_path.suffix.lower() in ['.mp4', '.avi', '.mov'] if is_video else ['.jpg', '.png', '.jpeg']:
                    self.samples.append((file_path, 0))
        
        # Fake samples (label 1)
        fake_dir = self.data_dir / 'fake'
        if fake_dir.exists():
            for file_path in fake_dir.glob('*'):
                if file_path.suffix.lower() in ['.mp4', '.avi', '.mov'] if is_video else ['.jpg', '.png', '.jpeg']:
                    self.samples.append((file_path, 1))
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        try:
            if self.is_video:
                tensor = self.preprocessor.process_video(file_path)
            else:
                tensor = self.preprocessor.process_image(file_path)
            
            # Normalize
            tensor = normalize_tensor(tensor)
            
            return tensor.squeeze(0), label
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a dummy tensor
            if self.is_video:
                return torch.zeros(16, 3, 224, 224), label
            else:
                return torch.zeros(3, 224, 224), label


class Trainer:
    """Training manager for IvyFake detector"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer and loss
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(
        self,
        num_epochs: int,
        save_dir: Path,
        save_every: int = 5
    ):
        """Complete training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.2f}%")
            
            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            if self.val_loader:
                print(f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.2f}%")
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics['loss'])
                
                # Save best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    torch.save(
                        self.model.state_dict(),
                        save_dir / 'best_model.pth'
                    )
                    print("Saved best model!")
            
            # Periodic save
            if epoch % save_every == 0:
                torch.save(
                    self.model.state_dict(),
                    save_dir / f'checkpoint_epoch_{epoch}.pth'
                )
        
        # Save final model
        torch.save(
            self.model.state_dict(),
            save_dir / 'final_model.pth'
        )
        
        # Save training history
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\nTraining completed!")


def main():
    """Main training function"""
    # Configuration
    DATA_DIR = Path(__file__).parent.parent / 'data' / 'train'
    SAVE_DIR = Path(__file__).parent / 'weights'
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize preprocessor
    preprocessor = VideoPreprocessor(
        target_size=(224, 224),
        max_frames=16,
        use_face_detection=False,  # Set to True if you have face detection setup
        device=device
    )
    
    # Create datasets
    print("Loading training data...")
    train_dataset = VideoDataset(
        DATA_DIR,
        preprocessor,
        is_video=False  # Set to True for video training
    )
    
    if len(train_dataset) == 0:
        print("No training data found! Please add videos/images to data/train/real and data/train/fake")
        return
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Initialize model
    print("Initializing model...")
    model = SimplifiedIvyDetector()
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,  # Add validation loader if needed
        device=device,
        lr=LEARNING_RATE
    )
    
    # Train
    print("Starting training...")
    trainer.train(
        num_epochs=NUM_EPOCHS,
        save_dir=SAVE_DIR,
        save_every=5
    )


if __name__ == '__main__':
    main()