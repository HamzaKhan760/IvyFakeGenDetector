"""
Training script for Multi-Scale IvyFake detector
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.multiscale_detector import MultiScaleIvyDetector
from utils.multiscale_preprocessing import MultiScaleVideoPreprocessor, normalize_tensor


class MultiScaleVideoDataset(Dataset):
    """Dataset for multi-scale video classification"""
    
    def __init__(
        self,
        data_dir: Path,
        preprocessor: MultiScaleVideoPreprocessor,
        is_video: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.is_video = is_video
        
        # Load samples
        self.samples = []
        
        # Real samples
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for file_path in real_dir.glob('*'):
                if file_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                    self.samples.append((file_path, 0))
        
        # Fake samples
        fake_dir = self.data_dir / 'fake'
        if fake_dir.exists():
            for file_path in fake_dir.glob('*'):
                if file_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                    self.samples.append((file_path, 1))
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        try:
            # Process at multiple scales
            scales = self.preprocessor.process_video(file_path)
            
            # Normalize
            slow = normalize_tensor(scales['slow']).squeeze(0)
            medium = normalize_tensor(scales['medium']).squeeze(0)
            fast = normalize_tensor(scales['fast']).squeeze(0)
            
            return (slow, medium, fast), label
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return dummy tensors
            return (
                torch.zeros(5, 3, 224, 224),
                torch.zeros(10, 3, 224, 224),
                torch.zeros(20, 3, 224, 224)
            ), label


def train_multiscale():
    """Main training function"""
    
    # Config
    DATA_DIR = Path(__file__).parent.parent / 'data' / 'train'
    SAVE_DIR = Path(__file__).parent / 'weights'
    BATCH_SIZE = 4  # Smaller due to multiple scales
    NUM_EPOCHS = 15
    LEARNING_RATE = 5e-5
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize preprocessor
    preprocessor = MultiScaleVideoPreprocessor(
        target_size=(224, 224),
        slow_fps=0.5,
        medium_fps=1.0,
        fast_fps=2.0,
        max_duration=10,
        use_face_detection=False,
        device=device
    )
    
    # Create dataset
    print("Loading training data...")
    train_dataset = MultiScaleVideoDataset(DATA_DIR, preprocessor)
    
    if len(train_dataset) == 0:
        print("No training data found!")
        return
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Initialize model
    print("Initializing multi-scale model...")
    model = MultiScaleIvyDetector()
    model.to(device)
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training history
    history = {'train_loss': [], 'train_acc': []}
    
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 50)
        
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, ((slow, medium, fast), target) in enumerate(pbar):
            slow = slow.to(device)
            medium = medium.to(device)
            fast = fast.to(device)
            target = target.to(device)
            
            # Forward
            optimizer.zero_grad()
            output = model(slow, medium, fast)
            loss = criterion(output['logits'], target)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output['logits'].argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(accuracy)
        
        print(f"Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), SAVE_DIR / 'multiscale_best_model.pth')
            print("✅ Saved best model!")
        
        scheduler.step(avg_loss)
    
    # Save final model
    torch.save(model.state_dict(), SAVE_DIR / 'multiscale_final_model.pth')
    
    # Save history
    with open(SAVE_DIR / 'multiscale_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n✅ Training completed!")


if __name__ == '__main__':
    train_multiscale()