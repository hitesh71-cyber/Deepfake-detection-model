import streamlit as st
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import Dropout
import tensorflow as tf
import tempfile
import os
from PIL import Image
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="DeepFake Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 DeepFake Detection System")
st.markdown("Upload an image or video to detect if faces are real or synthetic")

# Custom FixedDropout implementation
class FixedDropout(Dropout):
    def get_config(self):
        return super().get_config()

# Register custom objects
tf.keras.utils.get_custom_objects().update({
    'FixedDropout': FixedDropout,
    'swish': swish
})

# Load model
@st.cache_resource
def load_trained_model():
    try:
        model_path = './tmp_checkpoint/best_model.h5'
        if os.path.exists(model_path):
            # Load with custom objects
            model = load_model(
                model_path, 
                custom_objects={
                    'FixedDropout': FixedDropout,
                    'swish': swish
                },
                compile=False
            )
            return model
        else:
            st.warning("⚠️ Trained model not found. Please train the model first using `python 03-train_cnn.py`")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize MTCNN detector
@st.cache_resource
def get_detector():
    return MTCNN()

def preprocess_face(face_image, target_size=128):
    """Preprocess face for model prediction"""
    face_resized = cv2.resize(face_image, (target_size, target_size))
    face_normalized = face_resized / 255.0
    return face_normalized

def detect_and_predict_faces(image_array, model, detector):
    """Detect faces and make predictions"""
    results = []
    
    # Convert BGR to RGB for MTCNN
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    detections = detector.detect_faces(image_rgb)
    
    if not detections:
        return None, "No faces detected in the image"
    
    for detection in detections:
        confidence = detection['confidence']
        
        # Skip low confidence detections
        if confidence < 0.95:
            continue
        
        # Extract face region with margin
        box = detection['box']
        x, y, w, h = box
        
        # Add 30% margin
        margin_x = int(w * 0.3)
        margin_y = int(h * 0.3)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image_array.shape[1], x + w + margin_x)
        y2 = min(image_array.shape[0], y + h + margin_y)
        
        face_crop = image_array[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            continue
        
        # Preprocess
        face_processed = preprocess_face(face_crop)
        
        # Predict
        prediction = model.predict(np.expand_dims(face_processed, axis=0), verbose=0)[0][0]
        
        results.append({
            'box': (x1, y1, x2, y2),
            'score': float(prediction),
            'is_real': prediction > 0.5
        })
    
    return results, "Success"

def draw_predictions(image, predictions):
    """Draw bounding boxes and labels on image"""
    image_marked = image.copy()
    
    for pred in predictions:
        x1, y1, x2, y2 = pred['box']
        score = pred['score']
        is_real = pred['is_real']
        
        # Color: green for real, red for fake
        color = (0, 255, 0) if is_real else (0, 0, 255)
        label = f"REAL {score:.2f}" if is_real else f"FAKE {1-score:.2f}"
        
        # Draw box
        cv2.rectangle(image_marked, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        cv2.putText(image_marked, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return image_marked

# Load model
model = load_trained_model()

if model is None:
    st.error("Cannot proceed without a trained model.")
else:
    detector = get_detector()
    
    # Create tabs
    tab1, tab2 = st.tabs(["📸 Image Upload", "🎬 Video Upload"])
    
    # TAB 1: Image Upload
    with tab1:
        st.header("Upload an Image")
        
        uploaded_image = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png', 'bmp'])
        
        if uploaded_image is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Display original
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Detect and predict
            with st.spinner("Analyzing image..."):
                predictions, message = detect_and_predict_faces(image, model, detector)
            
            if predictions:
                with col2:
                    st.subheader("Detection Results")
                    marked_image = draw_predictions(image, predictions)
                    st.image(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB), use_column_width=True)
                
                # Show detailed results
                st.subheader("Detailed Results")
                for i, pred in enumerate(predictions, 1):
                    score = pred['score']
                    label = "🟢 REAL" if pred['is_real'] else "🔴 FAKE"
                    confidence = max(score, 1 - score)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"Face {i}", label)
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    with col3:
                        st.metric("Score", f"{score:.4f}")
                    
                    st.progress(score)
            else:
                st.warning(f"⚠️ {message}")
    
    # TAB 2: Video Upload
    with tab2:
        st.header("Upload a Video")
        
        uploaded_video = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_video is not None:
            # Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                tmp_video_path = tmp_file.name
            
            try:
                st.info("Processing video... This may take a moment.")
                
                # Open video
                cap = cv2.VideoCapture(tmp_video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Process frames at intervals to avoid slowdown
                frame_interval = max(1, total_frames // 10)  # Sample 10 frames
                frame_count = 0
                all_predictions = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_interval == 0:
                        predictions, _ = detect_and_predict_faces(frame, model, detector)
                        if predictions:
                            all_predictions.extend(predictions)
                        
                        progress = frame_count / total_frames
                        progress_bar.progress(min(progress, 1.0))
                        status_text.text(f"Processed {frame_count}/{total_frames} frames")
                    
                    frame_count += 1
                
                cap.release()
                progress_bar.empty()
                status_text.empty()
                
                # Show results
                if all_predictions:
                    st.success(f"✅ Analysis complete! Found {len(all_predictions)} faces across {frame_count} frames")
                    
                    # Statistics
                    real_count = sum(1 for p in all_predictions if p['is_real'])
                    fake_count = len(all_predictions) - real_count
                    avg_score = np.mean([p['score'] for p in all_predictions])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Real Faces", real_count)
                    with col2:
                        st.metric("Fake Faces", fake_count)
                    with col3:
                        st.metric("Average Score", f"{avg_score:.4f}")
                    
                    # Detailed breakdown
                    st.subheader("Score Distribution")
                    scores = [p['score'] for p in all_predictions]
                    
                    # Create histogram using matplotlib
                    fig, ax = plt.subplots()
                    ax.hist(scores, bins=20, color='skyblue', edgecolor='black')
                    ax.set_xlabel('Realness Score')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Realness Scores')
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ No faces detected in the video")
            
            finally:
                # Cleanup
                if os.path.exists(tmp_video_path):
                    os.remove(tmp_video_path)
    
    # Info section
    st.divider()
    st.markdown("""
    ### How to Interpret Results
    - **Score closer to 1.0**: Image is more likely to be **REAL**
    - **Score closer to 0.0**: Image is more likely to be **FAKE**
    
    ### Model Details
    - **Architecture**: EfficientNet B0 with custom classification head
    - **Input Size**: 128×128 pixels
    - **Training**: Binary classification (Real vs Deepfake)
    - **Face Detection**: MTCNN with 95% confidence threshold
    """)
