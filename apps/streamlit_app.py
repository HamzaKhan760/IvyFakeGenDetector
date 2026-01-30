"""
Streamlit Web App for IvyFake Detector
Upload videos or images to detect AI-generated content
"""

import streamlit as st
import torch
import sys
from pathlib import Path
import tempfile
import time
import cv2
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.detector import SimplifiedIvyDetector
from utils.preprocessing import VideoPreprocessor, normalize_tensor


# Page configuration
st.set_page_config(
    page_title="IvyFake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model(weights_path=None):
    """Load the detector model (cached)"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimplifiedIvyDetector()
    
    if weights_path and Path(weights_path).exists():
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        st.success(f"✅ Loaded model from {weights_path}")
    else:
        st.warning("⚠️ No pretrained weights loaded. Model initialized with random weights.")
    
    model.to(device)
    model.eval()
    return model, device


def get_video_thumbnail(video_path, frame_idx=0):
    """Extract a thumbnail from video"""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    return None


def process_video(video_path, model, device, preprocessor):
    """Process video and return prediction"""
    try:
        # Preprocess video
        with st.spinner("Extracting and analyzing frames..."):
            video_tensor = preprocessor.process_video(video_path)
            video_tensor = normalize_tensor(video_tensor)
            
            # Handle single image as video with 1 frame
            if video_tensor.dim() == 4:  # (1, C, H, W)
                video_tensor = video_tensor.unsqueeze(1)  # (1, 1, C, H, W)
            
            video_tensor = video_tensor.to(device)
        
        # Get prediction
        with st.spinner("Running detection..."):
            with torch.no_grad():
                # Flatten if necessary for SimplifiedIvyDetector
                if video_tensor.dim() == 5:
                    batch_size, num_frames = video_tensor.shape[:2]
                    video_tensor = video_tensor.view(-1, *video_tensor.shape[2:])
                
                logits = model(video_tensor)
                
                # Average predictions across frames
                if batch_size > 1 or num_frames > 1:
                    probs = torch.softmax(logits, dim=-1).mean(dim=0, keepdim=True)
                else:
                    probs = torch.softmax(logits, dim=-1)
                
                pred = torch.argmax(probs, dim=-1)
                confidence = probs[0, pred].item()
        
        return pred.item(), confidence
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None, None


def process_image(image_path, model, device, preprocessor):
    """Process image and return prediction"""
    try:
        # Preprocess image
        with st.spinner("Analyzing image..."):
            image_tensor = preprocessor.process_image(image_path)
            image_tensor = normalize_tensor(image_tensor)
            image_tensor = image_tensor.to(device)
        
        # Get prediction
        with st.spinner("Running detection..."):
            with torch.no_grad():
                logits = model(image_tensor)
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1)
                confidence = probs[0, pred].item()
        
        return pred.item(), confidence
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None


def main():
    # Title and description
    st.title("🔍 IvyFake Detector")
    st.markdown("""
    ### Explainable AI-Generated Content Detection
    Upload a video or image to detect whether it's authentic or AI-generated.
    
    Based on the **IvyFake** framework - a unified approach for detecting AI-generated images and videos.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model weights selection
        weights_dir = Path(__file__).parent.parent / 'training' / 'weights'
        weights_files = []
        if weights_dir.exists():
            weights_files = list(weights_dir.glob('*.pth'))
        
        if weights_files:
            weights_options = ['None'] + [f.name for f in weights_files]
            selected_weights = st.selectbox(
                "Model Weights",
                weights_options,
                help="Select pretrained weights or 'None' for random initialization"
            )
            
            if selected_weights != 'None':
                weights_path = weights_dir / selected_weights
            else:
                weights_path = None
        else:
            st.info("No pretrained weights found in training/weights/")
            weights_path = None
        
        # Processing options
        st.subheader("Processing Options")
        use_face_detection = st.checkbox(
            "Use Face Detection",
            value=False,
            help="Crop and focus on faces (requires MTCNN)"
        )
        
        max_frames = st.slider(
            "Max Frames (Video)",
            min_value=4,
            max_value=32,
            value=16,
            step=4,
            help="Number of frames to extract from video"
        )
        
        # Information
        st.markdown("---")
        st.markdown("""
        **About IvyFake:**
        - 🎯 Detects AI-generated images & videos
        - 🧠 Unified vision-language model
        - 📊 Analyzes temporal & spatial artifacts
        - 💡 Provides explainable results
        
        [GitHub](https://github.com/Pi3AI/IvyFake) | 
        [Paper](https://arxiv.org/abs/2501.xxxxx)
        """)
    
    # Load model
    model, device = load_model(weights_path)
    
    # Initialize preprocessor
    preprocessor = VideoPreprocessor(
        target_size=(224, 224),
        max_frames=max_frames,
        use_face_detection=use_face_detection,
        device=device
    )
    
    # File upload
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload a video or image",
        type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'],
        help="Supported formats: MP4, AVI, MOV (video) | JPG, JPEG, PNG (image)"
    )
    
    if uploaded_file is not None:
        # Determine file type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        is_video = file_extension in ['mp4', 'avi', 'mov']
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Display uploaded content
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Uploaded Content")
            if is_video:
                # Show video thumbnail
                thumbnail = get_video_thumbnail(tmp_path, frame_idx=10)
                if thumbnail:
                    st.image(thumbnail, caption="Video Thumbnail", use_container_width=True)
                st.video(tmp_path)
            else:
                st.image(tmp_path, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("🎯 Detection Results")
            
            # Run detection
            if st.button("🔍 Analyze Content", type="primary", use_container_width=True):
                start_time = time.time()
                
                if is_video:
                    prediction, confidence = process_video(tmp_path, model, device, preprocessor)
                else:
                    prediction, confidence = process_image(tmp_path, model, device, preprocessor)
                
                elapsed_time = time.time() - start_time
                
                if prediction is not None:
                    # Display results
                    st.markdown("---")
                    
                    if prediction == 0:
                        st.success("✅ **AUTHENTIC**")
                        result_color = "green"
                    else:
                        st.error("⚠️ **AI-GENERATED**")
                        result_color = "red"
                    
                    # Confidence meter
                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%",
                        help="Model's confidence in the prediction"
                    )
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # Additional info
                    st.info(f"⏱️ Analysis completed in {elapsed_time:.2f} seconds")
                    
                    # Explanation placeholder
                    with st.expander("📝 Detection Reasoning"):
                        st.markdown("""
                        **Analyzed Artifacts:**
                        - Spatial inconsistencies in pixel patterns
                        - Temporal coherence across frames (video)
                        - Texture and edge anomalies
                        - Facial feature irregularities (if face detected)
                        
                        *Note: Detailed natural language explanations require the full IvyXDetector model.*
                        """)
        
        # Cleanup
        Path(tmp_path).unlink()
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Upload a video or image to get started")
        
        # Example results
        with st.expander("📊 Example Results"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Authentic Video**")
                st.success("✅ Confidence: 95.3%")
                st.caption("Real person speaking")
            
            with col2:
                st.markdown("**Deepfake Video**")
                st.error("⚠️ Confidence: 92.1%")
                st.caption("AI-generated face swap")
            
            with col3:
                st.markdown("**Synthetic Image**")
                st.error("⚠️ Confidence: 87.6%")
                st.caption("Diffusion model generated")


if __name__ == "__main__":
    main()