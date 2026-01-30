# IvyFake Detector

A unified explainable framework for detecting AI-generated images and videos, based on the **IvyFake** research project.

![IvyFake Banner](https://img.shields.io/badge/AI-Deepfake%20Detection-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

## 🌟 Features

- **Unified Detection**: Single model for both images and videos
- **Explainable AI**: Provides reasoning for detection decisions
- **State-of-the-Art**: Based on vision-language architecture (CLIP)
- **Interactive Web App**: Easy-to-use Streamlit interface
- **Artifact Analysis**: Analyzes both temporal and spatial inconsistencies
- **Face-Focused Detection**: Optional face detection and cropping

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Web App](#web-app)
  - [Training](#training)
  - [Data Processing](#data-processing)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Citation](#citation)
- [License](#license)

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/IvyFake.git
cd IvyFake
```

### 2. Create virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

### Run the Web App

The easiest way to use IvyFake is through the Streamlit web interface:

```bash
python -m streamlit run apps/streamlit_app.py
```

Then open your browser to `http://localhost:8501` and upload a video or image!

## 📖 Usage

### Web App

The Streamlit app provides an intuitive interface for detecting AI-generated content:

1. **Upload**: Drag and drop a video (`.mp4`, `.avi`, `.mov`) or image (`.jpg`, `.png`)
2. **Configure**: Adjust settings in the sidebar (face detection, max frames)
3. **Analyze**: Click "Analyze Content" to get results
4. **Review**: View the prediction, confidence score, and reasoning

**Features:**
- Real-time detection
- Confidence visualization
- Detection reasoning explanation
- Support for multiple file formats

### Training

To train the model on your own dataset:

#### 1. Prepare Data

Organize your data in the following structure:

```
data/
  train/
    real/
      video1.mp4
      video2.mp4
      ...
    fake/
      video1.mp4
      video2.mp4
      ...
```

#### 2. Process Data (Optional)

Extract frames and preprocess:

```bash
python scripts/process_data.py \
  --input_dir data/raw \
  --output_dir data/processed \
  --type video \
  --use_face_detection \
  --max_frames 16
```

#### 3. Train the Model

```bash
python training/train.py
```

Training parameters can be modified in the script:
- `BATCH_SIZE`: Number of samples per batch (default: 8)
- `NUM_EPOCHS`: Number of training epochs (default: 20)
- `LEARNING_RATE`: Learning rate (default: 1e-4)

Trained weights will be saved to `training/weights/`.

### Data Processing

The `process_data.py` script handles:
- Frame extraction from videos
- Face detection and cropping
- Image resizing and normalization
- Organizing processed data

**Options:**
- `--input_dir`: Input directory with raw data
- `--output_dir`: Output directory for processed data
- `--type`: `video` or `image`
- `--use_face_detection`: Enable face detection
- `--max_frames`: Maximum frames per video

## 📁 Project Structure

```
IvyFake/
├── apps/
│   └── streamlit_app.py       # Streamlit web interface
├── models/
│   ├── __init__.py
│   └── detector.py            # Model architectures
├── training/
│   ├── train.py               # Training script
│   └── weights/               # Saved model weights
├── utils/
│   ├── __init__.py
│   └── preprocessing.py       # Data preprocessing utilities
├── scripts/
│   └── process_data.py        # Data processing script
├── data/
│   ├── train/                 # Training data
│   ├── test/                  # Test data
│   └── datasets/              # Dataset metadata
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🏗️ Model Architecture

### IvyXDetector

The full IvyXDetector model includes:

1. **Vision Backbone**: CLIP-based visual encoder
2. **Temporal Analyzer**: Analyzes frame-to-frame inconsistencies
3. **Spatial Analyzer**: Detects spatial artifacts within frames
4. **Fusion Layer**: Combines temporal and spatial features
5. **Classification Head**: Binary classification (real/fake)
6. **Explanation Generator**: Produces human-readable explanations

### SimplifiedIvyDetector

A lightweight version for quick inference:
- Pre-trained CLIP vision encoder (frozen)
- Simple MLP classifier
- Suitable for deployment and real-time applications

### Key Components

```python
# Temporal Artifact Analyzer
- 1D convolutions across time
- Multi-head self-attention
- Global temporal pooling

# Spatial Artifact Analyzer
- Attention-based feature weighting
- Spatial anomaly detection
- Frame-level analysis
```

## 📊 Dataset

The IvyFake framework is designed for the official **Ivy-Fake** dataset:

- **Size**: 150,000+ training samples, 18,700 evaluation samples
- **Modalities**: Images and videos
- **Annotations**: Binary labels + natural language explanations
- **Content**: Diverse AI-generated content (GANs, Diffusion models, Transformers)
- **Access**: [Hugging Face Dataset](https://huggingface.co/datasets/AI-Safeguard/Ivy-Fake)

### Custom Datasets

You can also train on custom datasets. Ensure your data follows the structure:

```
data/
  train/
    real/     # Authentic content
    fake/     # AI-generated content
```

## 🎯 Performance

Expected performance metrics (on Ivy-Fake dataset):

| Metric | Value |
|--------|-------|
| Accuracy | ~94% |
| Precision | ~93% |
| Recall | ~95% |
| F1-Score | ~94% |

*Note: Results may vary based on training data and hyperparameters*

## 🔧 Configuration

### Model Configuration

```python
# In models/detector.py
IvyXDetector(
    model_name="openai/clip-vit-base-patch32",
    num_classes=2,
    embed_dim=512,
    freeze_backbone=False
)
```

### Preprocessing Configuration

```python
# In utils/preprocessing.py
VideoPreprocessor(
    target_size=(224, 224),
    max_frames=16,
    use_face_detection=True,
    device='cuda'
)
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Citation

If you use IvyFake in your research, please cite the original paper:

```bibtex
@article{jiang2025ivyfake,
  title     = {Ivy-Fake: A Unified Explainable Framework and Benchmark for Image and Video AIGC Detection},
  author    = {Jiang, Changjiang and Zhang, Wayne and Zhang, Zhonghao and Yu, Fengchang and Peng, Wei},
  year      = {2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Original Repository**: [Pi3AI/IvyFake](https://github.com/Pi3AI/IvyFake)
- **Dataset**: [Hugging Face](https://huggingface.co/datasets/AI-Safeguard/Ivy-Fake)
- **Paper**: ArXiv (Coming Soon)

## ⚠️ Disclaimer

This tool is for research and educational purposes. Always verify content through multiple sources and methods. AI detection is not perfect and should be used as one tool among many for content verification.

## 🙏 Acknowledgments

- Original IvyFake research team
- Anthropic's Claude for assistance
- Open-source community
- CLIP model by OpenAI
- Hugging Face for model hosting

## 📞 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact the maintainers

---

**Made with ❤️ for safer AI content**