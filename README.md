# DeepFake Detection System

A machine learning-based toolkit for identifying synthetic and manipulated facial content in images and videos.

## About This Project

This system uses deep neural networks to detect whether faces in images or videos are authentic or artificially generated. The project combines computer vision for face detection with a convolutional neural network trained to classify faces as real or deepfake.

The workflow processes raw video files, automatically extracts facial regions, and trains a classification model that can identify manipulated content with reasonable accuracy.

## How It Works

The system operates in two phases:

**Training Phase:**
1. Extract frames from source videos
2. Detect and crop all faces using MTCNN (Multi-task Cascaded Convolutional Networks)
3. Balance the dataset (since fake samples typically outnumber real ones)
4. Train an EfficientNet-based neural network on the prepared data

**Prediction Phase:**
1. Accept image or video uploads through a web interface
2. Extract and detect faces in the input
3. Preprocess faces to match training specifications (128×128 pixels)
4. Run the trained model to classify each face
5. Display confidence scores and predictions

## Project Structure

```
├── 00-convert_video_to_image.py      # Extract frames from videos
├── 01a-crop_faces_with_mtcnn.py      # Detect and crop facial regions
├── 01b-crop_faces_with_azure-vision-api.py  # Alternative: cloud-based face detection
├── 02-prepare_fake_real_dataset.py   # Balance and organize training data
├── 03-train_cnn.py                   # Train the classification model
├── streamlit_app.py                  # Web interface for predictions
├── requirements.txt                  # Python dependencies
└── split_dataset/                    # Training/validation/test splits
    ├── train/
    ├── val/
    └── test/
```

## Getting Started

### Requirements

- Python 3.8+
- TensorFlow 2.x
- GPU support (optional but recommended)

### Training a Model

Run the pipeline scripts in sequence:

```bash
# Step 1: Extract frames from your video files
python 00-convert_video_to_image.py

# Step 2: Detect and crop faces
python 01a-crop_faces_with_mtcnn.py

# Step 3: Prepare training data
python 02-prepare_fake_real_dataset.py

# Step 4: Train the model
python 03-train_cnn.py
```

Place your videos in `train_sample_videos/` directory with a `metadata.json` file specifying labels (REAL or FAKE).

### Using the Web Interface

Once you have a trained model, run:

```bash
streamlit run streamlit_app.py
```

Then open your browser to the displayed URL and upload images or videos for analysis.

## Model Architecture

The classification model uses:
- **Backbone:** EfficientNet B0 (pre-trained weights optional)
- **Input:** 128×128 RGB images
- **Hidden Layers:** Global max pooling followed by two dense layers (512 and 128 units)
- **Output:** Single sigmoid neuron (probability of being real)

The model produces a score between 0 and 1:
- **Closer to 1:** Image likely authentic
- **Closer to 0:** Image likely synthetic/deepfake

## Face Detection

The system uses MTCNN for face localization:
- Detects faces with 95% confidence threshold
- Adds 30% margin around detected bounding boxes
- Handles multiple faces per frame independently

## Data Processing

Smart image resizing based on quality:
- Videos < 300px wide: 2× upscale
- Videos 300-1000px: 1× (no resize)
- Videos 1000-1900px: 0.5× downscale
- Videos > 1900px: 0.33× downscale

This balances accuracy with processing speed and memory usage.

## Common Issues & Fixes

**GPU not detected:**
Add a check before setting memory growth in TensorFlow

**Missing packages:**
Run `pip install -r requirements.txt` to install dependencies

**Model training slow:**
Consider using GPU acceleration or reducing dataset size for initial testing

## Future Enhancements

- Support for video-level predictions (temporal analysis)
- Real-time webcam analysis
- Batch processing capabilities
- Mobile app version
- Improved model accuracy with larger datasets