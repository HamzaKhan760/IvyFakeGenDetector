"""
Benchmarking script for IvyFake detector
Evaluate model accuracy on labeled test datasets
"""

import sys
import argparse
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import json
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

from models.detector import SimplifiedIvyDetector
from utils.preprocessing import VideoPreprocessor, normalize_tensor


class Benchmarker:
    """Benchmark IvyFake detector on test datasets"""
    
    def __init__(
        self,
        model_path: str,
        test_data_dir: Path,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_face_detection: bool = False,
        max_frames: int = 16
    ):
        self.device = device
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = SimplifiedIvyDetector()
        
        if model_path and Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            print("✅ Model loaded successfully")
        else:
            print("⚠️ No model weights provided, using random initialization")
        
        self.model.to(device)
        self.model.eval()
        
        # Initialize preprocessor
        self.preprocessor = VideoPreprocessor(
            target_size=(224, 224),
            max_frames=max_frames,
            use_face_detection=use_face_detection,
            device=device
        )
        
        # Load test data
        self.test_data_dir = Path(test_data_dir)
        self.test_samples = self._load_test_samples()
        
    def _load_test_samples(self):
        """Load test samples with labels"""
        samples = []
        
        # Real samples (label 0)
        real_dir = self.test_data_dir / 'real'
        if real_dir.exists():
            for file_path in real_dir.glob('*'):
                if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.jpg', '.jpeg', '.png']:
                    samples.append((file_path, 0, 'real'))
        
        # Fake samples (label 1)
        fake_dir = self.test_data_dir / 'fake'
        if fake_dir.exists():
            for file_path in fake_dir.glob('*'):
                if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.jpg', '.jpeg', '.png']:
                    samples.append((file_path, 1, 'fake'))
        
        print(f"\nLoaded {len(samples)} test samples:")
        print(f"  - Real: {sum(1 for s in samples if s[1] == 0)}")
        print(f"  - Fake: {sum(1 for s in samples if s[1] == 1)}")
        
        return samples
    
    def predict(self, file_path: Path):
        """Make prediction on a single file"""
        try:
            # Determine if video or image
            is_video = file_path.suffix.lower() in ['.mp4', '.avi', '.mov']
            
            # Preprocess
            if is_video:
                tensor = self.preprocessor.process_video(file_path)
            else:
                tensor = self.preprocessor.process_image(file_path)
            
            tensor = normalize_tensor(tensor)
            
            # Handle dimensions
            if tensor.dim() == 5:  # Video
                batch_size, num_frames = tensor.shape[:2]
                tensor = tensor.view(-1, *tensor.shape[2:])
            
            tensor = tensor.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                logits = self.model(tensor)
                
                if tensor.shape[0] > 1:
                    probs = torch.softmax(logits, dim=-1).mean(dim=0, keepdim=True)
                else:
                    probs = torch.softmax(logits, dim=-1)
                
                pred = torch.argmax(probs, dim=-1)
                confidence = probs[0, pred].item()
            
            return pred.item(), confidence
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None, None
    
    def run_benchmark(self, save_dir: Path = None):
        """Run full benchmark evaluation"""
        print(f"\n{'='*60}")
        print("Running Benchmark")
        print(f"{'='*60}\n")
        
        y_true = []
        y_pred = []
        confidences = []
        results_detailed = []
        
        for file_path, true_label, label_name in tqdm(self.test_samples, desc="Evaluating"):
            pred, confidence = self.predict(file_path)
            
            if pred is not None:
                y_true.append(true_label)
                y_pred.append(pred)
                confidences.append(confidence)
                
                results_detailed.append({
                    'file': file_path.name,
                    'true_label': label_name,
                    'predicted_label': 'real' if pred == 0 else 'fake',
                    'confidence': confidence,
                    'correct': pred == true_label
                })
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Print results
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}\n")
        print(f"Accuracy:  {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall:    {recall*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")
        print(f"\nAverage Confidence: {np.mean(confidences)*100:.2f}%")
        
        print(f"\n{'='*60}")
        print("Confusion Matrix:")
        print(f"{'='*60}")
        print(f"                 Predicted")
        print(f"              Real    Fake")
        print(f"Actual Real   {cm[0][0]:4d}    {cm[0][1]:4d}")
        print(f"       Fake   {cm[1][0]:4d}    {cm[1][1]:4d}")
        
        print(f"\n{'='*60}")
        print("Classification Report:")
        print(f"{'='*60}")
        print(classification_report(
            y_true, y_pred,
            target_names=['Real', 'Fake'],
            digits=4
        ))
        
        # Save results
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'avg_confidence': float(np.mean(confidences)),
                'confusion_matrix': cm.tolist(),
                'total_samples': len(y_true)
            }
            
            with open(save_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save detailed results
            df = pd.DataFrame(results_detailed)
            df.to_csv(save_dir / 'detailed_results.csv', index=False)
            
            # Plot confusion matrix
            self._plot_confusion_matrix(cm, save_dir / 'confusion_matrix.png')
            
            # Plot confidence distribution
            self._plot_confidence_distribution(
                results_detailed,
                save_dir / 'confidence_distribution.png'
            )
            
            print(f"\n✅ Results saved to {save_dir}")
        
        return metrics
    
    def _plot_confusion_matrix(self, cm, save_path):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self, results, save_path):
        """Plot confidence distribution"""
        df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Correct vs Incorrect
        correct = df[df['correct']]['confidence']
        incorrect = df[~df['correct']]['confidence']
        
        axes[0].hist(correct, bins=20, alpha=0.7, label='Correct', color='green')
        axes[0].hist(incorrect, bins=20, alpha=0.7, label='Incorrect', color='red')
        axes[0].set_xlabel('Confidence')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Confidence Distribution: Correct vs Incorrect')
        axes[0].legend()
        
        # Real vs Fake
        real = df[df['true_label'] == 'real']['confidence']
        fake = df[df['true_label'] == 'fake']['confidence']
        
        axes[1].hist(real, bins=20, alpha=0.7, label='Real', color='blue')
        axes[1].hist(fake, bins=20, alpha=0.7, label='Fake', color='orange')
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Confidence Distribution: Real vs Fake')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark IvyFake detector")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model weights'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        default='data/test',
        help='Path to test data directory'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='benchmark_results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--use_face_detection',
        action='store_true',
        help='Use face detection'
    )
    parser.add_argument(
        '--max_frames',
        type=int,
        default=16,
        help='Maximum frames for videos'
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmarker = Benchmarker(
        model_path=args.model,
        test_data_dir=Path(args.test_data),
        device=args.device,
        use_face_detection=args.use_face_detection,
        max_frames=args.max_frames
    )
    
    benchmarker.run_benchmark(save_dir=Path(args.save_dir))


if __name__ == '__main__':
    main()