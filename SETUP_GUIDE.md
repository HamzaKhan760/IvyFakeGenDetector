# IvyFake Setup Guide

## Quick Setup (5 minutes)

### 1. System Requirements

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- GPU with CUDA support (optional but recommended)
- 2GB+ free disk space

### 2. Installation Steps

#### Step 1: Clone or extract the repository

```bash
cd IvyFake
```

#### Step 2: Create virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Step 3: Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- PyTorch and torchvision
- Transformers (for CLIP model)
- OpenCV (for video processing)
- Streamlit (for web app)
- facenet-pytorch (for face detection)
- Other utilities

**Note:** If you encounter issues with PyTorch, visit [pytorch.org](https://pytorch.org/get-started/locally/) for platform-specific installation commands.

### 3. Quick Test

Test if everything is working:

```bash
python -c "import torch; import streamlit; import cv2; import transformers; print('✅ All dependencies loaded successfully!')"
```

## Running the Application

### Option 1: Web App (Recommended for beginners)

```bash
streamlit run apps/streamlit_app.py
```

Then open your browser to: `http://localhost:8501`

### Option 2: Command Line

```bash
python scripts/inference.py path/to/video.mp4 --device cpu
```

Available options:
- `--weights`: Path to trained model weights
- `--device`: `cpu` or `cuda`
- `--use_face_detection`: Enable face detection
- `--max_frames`: Number of frames to analyze

## Training Your Own Model

### Step 1: Prepare Dataset

Organize your data:

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

### Step 2: (Optional) Process Data

```bash
python scripts/process_data.py \
  --input_dir data/raw \
  --output_dir data/processed \
  --type video
```

### Step 3: Train

```bash
python training/train.py
```

Weights will be saved to `training/weights/`

## Troubleshooting

### Common Issues

#### 1. "No module named 'torch'"

**Solution:** Install PyTorch
```bash
pip install torch torchvision
```

#### 2. "CUDA out of memory"

**Solution:** Use CPU or reduce batch size
- Set `--device cpu` for inference
- Reduce `BATCH_SIZE` in `training/train.py`

#### 3. "Cannot find model weights"

**Solution:** 
- The model will work with random initialization but won't be accurate
- Train your own model or download pretrained weights
- In the web app, select "None" for weights to test functionality

#### 4. "Face detection not working"

**Solution:**
- Set `use_face_detection=False` in preprocessor
- Or install required dependencies:
```bash
pip install facenet-pytorch
```

#### 5. Streamlit app won't start

**Solution:**
```bash
pip install --upgrade streamlit
streamlit run apps/streamlit_app.py
```

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed: `pip list`
2. Verify Python version: `python --version` (should be 3.8+)
3. Try running with CPU: `--device cpu`
4. Check the error message carefully
5. Open an issue on GitHub with:
   - Error message
   - Python version
   - Operating system
   - Steps to reproduce

## Performance Optimization

### For Faster Inference

1. **Use GPU:**
   ```bash
   --device cuda
   ```

2. **Reduce frames:**
   ```bash
   --max_frames 8
   ```

3. **Disable face detection:**
   Remove `--use_face_detection` flag

### For Better Accuracy

1. **Use more frames:**
   ```bash
   --max_frames 32
   ```

2. **Enable face detection:**
   ```bash
   --use_face_detection
   ```

3. **Use trained weights:**
   ```bash
   --weights training/weights/best_model.pth
   ```

## Next Steps

1. **Test with sample data:** Try the web app with test videos/images
2. **Train on your data:** Prepare your dataset and train the model
3. **Experiment:** Try different settings and configurations
4. **Contribute:** Improve the codebase and share your findings

## Additional Resources

- **PyTorch Tutorials:** https://pytorch.org/tutorials/
- **Streamlit Docs:** https://docs.streamlit.io/
- **Original IvyFake Paper:** Check the README for the link
- **CLIP Model:** https://github.com/openai/CLIP

## System Architecture

```
┌─────────────────────────────────────────┐
│         Input (Video/Image)             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Video Preprocessor                 │
│  • Frame extraction                     │
│  • Face detection (optional)            │
│  • Resizing & normalization             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      IvyFake Detector Model             │
│  • CLIP vision encoder                  │
│  • Temporal/Spatial analyzers           │
│  • Classification head                  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    Output (Prediction + Confidence)     │
│  • AUTHENTIC or AI-GENERATED            │
│  • Confidence score (0-100%)            │
└─────────────────────────────────────────┘
```

---

**Ready to detect deepfakes? Start with the web app!**

```bash
streamlit run apps/streamlit_app.py
```