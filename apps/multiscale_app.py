"""
Multi-Scale Temporal Analysis Streamlit App
Compare standard IvyFake vs Multi-Scale version
"""

import streamlit as st
import torch
import sys
from pathlib import Path
import tempfile
import time
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from models.detector import SimplifiedIvyDetector
from models.multiscale_detector import MultiScaleIvyDetector
from utils.preprocessing import VideoPreprocessor, normalize_tensor
from utils.multiscale_preprocessing import MultiScaleVideoPreprocessor


st.set_page_config(
    page_title="IvyFake Multi-Scale Detector",
    page_icon="🔬",
    layout="wide"
)


@st.cache_resource
def load_models(standard_weights=None, multiscale_weights=None):
    """Load both standard and multi-scale models"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Standard model
    standard_model = SimplifiedIvyDetector()
    if standard_weights and Path(standard_weights).exists():
        state_dict = torch.load(standard_weights, map_location=device)
        standard_model.load_state_dict(state_dict)
    standard_model.to(device)
    standard_model.eval()
    
    # Multi-scale model
    multiscale_model = MultiScaleIvyDetector()
    if multiscale_weights and Path(multiscale_weights).exists():
        state_dict = torch.load(multiscale_weights, map_location=device)
        multiscale_model.load_state_dict(state_dict)
    multiscale_model.to(device)
    multiscale_model.eval()
    
    return standard_model, multiscale_model, device


def process_standard(video_path, model, device, preprocessor):
    """Process with standard IvyFake"""
    try:
        video_tensor = preprocessor.process_video(video_path)
        video_tensor = normalize_tensor(video_tensor)
        
        if video_tensor.dim() == 5:
            batch_size, num_frames = video_tensor.shape[:2]
            video_tensor = video_tensor.view(-1, *video_tensor.shape[2:])
        
        video_tensor = video_tensor.to(device)
        
        with torch.no_grad():
            logits = model(video_tensor)
            
            if video_tensor.shape[0] > 1:
                probs = torch.softmax(logits, dim=-1).mean(dim=0, keepdim=True)
            else:
                probs = torch.softmax(logits, dim=-1)
            
            pred = torch.argmax(probs, dim=-1)
            confidence = probs[0, pred].item()
        
        return pred.item(), confidence
    except Exception as e:
        st.error(f"Standard processing error: {e}")
        return None, None


def process_multiscale(video_path, model, device, preprocessor):
    """Process with multi-scale IvyFake"""
    try:
        # Extract at multiple scales
        scales = preprocessor.process_video(video_path)
        
        # Normalize each scale
        slow = normalize_tensor(scales['slow']).to(device)
        medium = normalize_tensor(scales['medium']).to(device)
        fast = normalize_tensor(scales['fast']).to(device)
        
        with torch.no_grad():
            output = model(slow, medium, fast)
            logits = output['logits']
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            confidence = probs[0, pred].item()
        
        return pred.item(), confidence, scales
    except Exception as e:
        st.error(f"Multi-scale processing error: {e}")
        return None, None, None


def main():
    st.title("🔬 IvyFake Multi-Scale Temporal Analysis")
    st.markdown("""
    ### Compare Detection Methods
    **Standard IvyFake:** Analyzes video at single frame rate (1 fps)  
    **Multi-Scale IvyFake:** Analyzes at multiple rates (0.5, 1.0, 2.0 fps) for better temporal artifact detection
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        detection_mode = st.radio(
            "Detection Mode",
            ["Standard IvyFake", "Multi-Scale IvyFake", "Compare Both"],
            help="Choose detection method"
        )
        
        # Weights selection
        weights_dir = Path(__file__).parent.parent / 'training' / 'weights'
        
        standard_weights = None
        multiscale_weights = None
        
        if weights_dir.exists():
            weight_files = list(weights_dir.glob('*.pth'))
            if weight_files:
                weight_options = ['None'] + [f.name for f in weight_files]
                
                if detection_mode in ["Standard IvyFake", "Compare Both"]:
                    std_weight = st.selectbox("Standard Weights", weight_options, key="std")
                    if std_weight != 'None':
                        standard_weights = weights_dir / std_weight
                
                if detection_mode in ["Multi-Scale IvyFake", "Compare Both"]:
                    ms_weight = st.selectbox("Multi-Scale Weights", weight_options, key="ms")
                    if ms_weight != 'None':
                        multiscale_weights = weights_dir / ms_weight
        
        # Processing options
        st.subheader("Processing Options")
        use_face_detection = st.checkbox("Use Face Detection", value=False)
        
        if detection_mode == "Multi-Scale IvyFake" or detection_mode == "Compare Both":
            st.subheader("Multi-Scale Settings")
            slow_fps = st.slider("Slow FPS", 0.25, 1.0, 0.5, 0.25)
            medium_fps = st.slider("Medium FPS", 0.5, 2.0, 1.0, 0.5)
            fast_fps = st.slider("Fast FPS", 1.0, 4.0, 2.0, 0.5)
        
        st.markdown("---")
        st.markdown("""
        **Multi-Scale Benefits:**
        - 🐌 Slow FPS: Detects long-term inconsistencies
        - 🚶 Medium FPS: Standard temporal analysis  
        - 🏃 Fast FPS: Captures micro-expressions & quick changes
        """)
    
    # Load models
    standard_model, multiscale_model, device = load_models(
        standard_weights, multiscale_weights
    )
    
    # Initialize preprocessors
    standard_preprocessor = VideoPreprocessor(
        target_size=(224, 224),
        max_frames=16,
        use_face_detection=use_face_detection,
        device=device
    )
    
    if detection_mode == "Multi-Scale IvyFake" or detection_mode == "Compare Both":
        multiscale_preprocessor = MultiScaleVideoPreprocessor(
            target_size=(224, 224),
            slow_fps=slow_fps,
            medium_fps=medium_fps,
            fast_fps=fast_fps,
            max_duration=10,
            use_face_detection=use_face_detection,
            device=device
        )
    
    # File upload
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload a video",
        type=['mp4', 'avi', 'mov'],
        help="Upload video for deepfake detection"
    )
    
    if uploaded_file:
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Display video
        st.video(tmp_path)
        
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            results = {}
            
            if detection_mode == "Standard IvyFake" or detection_mode == "Compare Both":
                with st.spinner("Running Standard IvyFake..."):
                    start = time.time()
                    pred, conf = process_standard(
                        tmp_path, standard_model, device, standard_preprocessor
                    )
                    elapsed = time.time() - start
                    results['Standard'] = {
                        'prediction': pred,
                        'confidence': conf,
                        'time': elapsed
                    }
            
            if detection_mode == "Multi-Scale IvyFake" or detection_mode == "Compare Both":
                with st.spinner("Running Multi-Scale IvyFake..."):
                    start = time.time()
                    pred, conf, scales = process_multiscale(
                        tmp_path, multiscale_model, device, multiscale_preprocessor
                    )
                    elapsed = time.time() - start
                    results['Multi-Scale'] = {
                        'prediction': pred,
                        'confidence': conf,
                        'time': elapsed,
                        'scales': scales
                    }
            
            # Display results
            st.markdown("---")
            st.markdown("### 🎯 Detection Results")
            
            if detection_mode == "Compare Both":
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Standard IvyFake")
                    if results['Standard']['prediction'] == 0:
                        st.success("✅ AUTHENTIC")
                    else:
                        st.error("⚠️ AI-GENERATED")
                    st.metric("Confidence", f"{results['Standard']['confidence']*100:.2f}%")
                    st.info(f"⏱️ Time: {results['Standard']['time']:.2f}s")
                
                with col2:
                    st.markdown("#### Multi-Scale IvyFake")
                    if results['Multi-Scale']['prediction'] == 0:
                        st.success("✅ AUTHENTIC")
                    else:
                        st.error("⚠️ AI-GENERATED")
                    st.metric("Confidence", f"{results['Multi-Scale']['confidence']*100:.2f}%")
                    st.info(f"⏱️ Time: {results['Multi-Scale']['time']:.2f}s")
                    
                    # Show scale info
                    with st.expander("📊 Multi-Scale Details"):
                        scales_data = results['Multi-Scale']['scales']
                        df = pd.DataFrame({
                            'Scale': ['Slow', 'Medium', 'Fast'],
                            'FPS': [slow_fps, medium_fps, fast_fps],
                            'Frames': [
                                scales_data['slow'].shape[1],
                                scales_data['medium'].shape[1],
                                scales_data['fast'].shape[1]
                            ]
                        })
                        st.dataframe(df, use_container_width=True)
            
            else:
                # Single mode
                mode = 'Standard' if detection_mode == "Standard IvyFake" else 'Multi-Scale'
                result = results[mode]
                
                if result['prediction'] == 0:
                    st.success("✅ **AUTHENTIC**")
                else:
                    st.error("⚠️ **AI-GENERATED**")
                
                st.metric("Confidence", f"{result['confidence']*100:.2f}%")
                st.info(f"⏱️ Analysis completed in {result['time']:.2f} seconds")
                
                if mode == 'Multi-Scale':
                    with st.expander("📊 Multi-Scale Details"):
                        scales_data = result['scales']
                        df = pd.DataFrame({
                            'Scale': ['Slow', 'Medium', 'Fast'],
                            'FPS': [slow_fps, medium_fps, fast_fps],
                            'Frames': [
                                scales_data['slow'].shape[1],
                                scales_data['medium'].shape[1],
                                scales_data['fast'].shape[1]
                            ]
                        })
                        st.dataframe(df, use_container_width=True)
        
        # Cleanup
        Path(tmp_path).unlink()
    
    else:
        st.info("👆 Upload a video to get started")


if __name__ == "__main__":
    main()