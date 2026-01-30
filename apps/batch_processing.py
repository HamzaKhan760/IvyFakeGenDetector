"""
Batch Processing App for IvyFake Detector
Upload multiple images/videos and get results for all
"""

import streamlit as st
import torch
import sys
from pathlib import Path
import tempfile
import time
import pandas as pd
import zipfile
from io import BytesIO
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.detector import SimplifiedIvyDetector
from utils.preprocessing import VideoPreprocessor, normalize_tensor


st.set_page_config(
    page_title="IvyFake Batch Processor",
    page_icon="📊",
    layout="wide"
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


def process_single_file(file_path, model, device, preprocessor, is_video=False):
    """Process a single file and return prediction"""
    try:
        # Preprocess
        if is_video:
            tensor = preprocessor.process_video(file_path)
        else:
            tensor = preprocessor.process_image(file_path)
        
        tensor = normalize_tensor(tensor)
        
        # Handle dimensions
        if tensor.dim() == 5:  # Video
            batch_size, num_frames = tensor.shape[:2]
            tensor = tensor.view(-1, *tensor.shape[2:])
        
        tensor = tensor.to(device)
        
        # Get prediction
        with torch.no_grad():
            logits = model(tensor)
            
            if tensor.shape[0] > 1:
                probs = torch.softmax(logits, dim=-1).mean(dim=0, keepdim=True)
            else:
                probs = torch.softmax(logits, dim=-1)
            
            pred = torch.argmax(probs, dim=-1)
            confidence = probs[0, pred].item()
        
        return pred.item(), confidence
        
    except Exception as e:
        return None, None


def main():
    st.title("📊 IvyFake Batch Processor")
    st.markdown("""
    Upload multiple images or videos at once and get detection results for all files.
    
    **Supported formats:** JPG, PNG, MP4, AVI, MOV
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model weights
        weights_dir = Path(__file__).parent.parent / 'training' / 'weights'
        weights_files = []
        if weights_dir.exists():
            weights_files = list(weights_dir.glob('*.pth'))
        
        if weights_files:
            weights_options = ['None'] + [f.name for f in weights_files]
            selected_weights = st.selectbox("Model Weights", weights_options)
            weights_path = weights_dir / selected_weights if selected_weights != 'None' else None
        else:
            st.info("No pretrained weights found")
            weights_path = None
        
        use_face_detection = st.checkbox("Use Face Detection", value=False)
        max_frames = st.slider("Max Frames (Video)", 4, 32, 16, 4)
        
        st.markdown("---")
        st.markdown("**Export Options**")
        export_format = st.radio("Export Format", ["CSV", "JSON", "Excel"])
    
    # Load model
    model, device = load_model(weights_path)
    
    # Initialize preprocessor
    preprocessor = VideoPreprocessor(
        target_size=(224, 224),
        max_frames=max_frames,
        use_face_detection=use_face_detection,
        device=device
    )
    
    st.markdown("---")
    
    # Upload method selection
    upload_method = st.radio(
        "Upload Method:",
        ["Multiple Files", "ZIP Archive"],
        horizontal=True
    )
    
    uploaded_files = []
    
    if upload_method == "Multiple Files":
        uploaded_files = st.file_uploader(
            "Upload images or videos",
            type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
            accept_multiple_files=True
        )
    else:
        zip_file = st.file_uploader(
            "Upload ZIP archive",
            type=['zip']
        )
        
        if zip_file:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir = Path(tmp_dir)
                
                # Extract ZIP
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)
                
                # Find all supported files
                supported_extensions = ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov']
                uploaded_files = []
                
                for ext in supported_extensions:
                    uploaded_files.extend(list(tmp_dir.rglob(f'*{ext}')))
                
                st.info(f"Found {len(uploaded_files)} files in archive")
    
    if uploaded_files and len(uploaded_files) > 0:
        st.markdown(f"### 📁 Files to Process: {len(uploaded_files)}")
        
        if st.button("🚀 Process All Files", type="primary", use_container_width=True):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                
                if hasattr(uploaded_file, 'name'):
                    file_name = uploaded_file.name
                    file_extension = file_name.split('.')[-1].lower()
                else:
                    file_name = uploaded_file.name
                    file_extension = uploaded_file.suffix.lower()[1:]
                
                status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {file_name}")
                
                is_video = file_extension in ['mp4', 'avi', 'mov']
                
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                    if hasattr(uploaded_file, 'read'):
                        tmp_file.write(uploaded_file.read())
                    else:
                        with open(uploaded_file, 'rb') as f:
                            tmp_file.write(f.read())
                    tmp_path = tmp_file.name
                
                # Process
                pred, confidence = process_single_file(
                    tmp_path, model, device, preprocessor, is_video
                )
                
                # Store results
                result_label = "AUTHENTIC" if pred == 0 else "AI-GENERATED"
                results.append({
                    'File Name': file_name,
                    'Type': 'Video' if is_video else 'Image',
                    'Prediction': result_label,
                    'Confidence': f"{confidence * 100:.2f}%" if confidence else "Error",
                    'Status': '✅' if pred is not None else '❌'
                })
                
                # Cleanup
                Path(tmp_path).unlink()
            
            total_time = time.time() - start_time
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Results")
            
            df = pd.DataFrame(results)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total_files = len(results)
            authentic_count = sum(1 for r in results if r['Prediction'] == 'AUTHENTIC')
            fake_count = sum(1 for r in results if r['Prediction'] == 'AI-GENERATED')
            error_count = sum(1 for r in results if r['Status'] == '❌')
            
            col1.metric("Total Files", total_files)
            col2.metric("Authentic", authentic_count, delta=f"{authentic_count/total_files*100:.1f}%")
            col3.metric("AI-Generated", fake_count, delta=f"{fake_count/total_files*100:.1f}%")
            col4.metric("Errors", error_count)
            
            st.info(f"⏱️ Total processing time: {total_time:.2f} seconds ({total_time/total_files:.2f}s per file)")
            
            # Results table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
            
            # Export options
            st.markdown("### 💾 Export Results")
            
            if export_format == "CSV":
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    "ivyfake_results.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            elif export_format == "JSON":
                json_str = df.to_json(orient='records', indent=2)
                st.download_button(
                    "📥 Download JSON",
                    json_str,
                    "ivyfake_results.json",
                    "application/json",
                    use_container_width=True
                )
            
            elif export_format == "Excel":
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Results')
                
                st.download_button(
                    "📥 Download Excel",
                    buffer.getvalue(),
                    "ivyfake_results.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    
    else:
        st.info("👆 Upload files to get started")
        
        # Instructions
        with st.expander("📖 Instructions"):
            st.markdown("""
            **How to use:**
            1. Choose upload method (Multiple Files or ZIP Archive)
            2. Upload your images/videos
            3. Click "Process All Files"
            4. View results and export
            
            **Tips:**
            - For large batches, use ZIP upload
            - Enable GPU in sidebar for faster processing
            - Export results as CSV/JSON/Excel for further analysis
            """)


if __name__ == "__main__":
    main()