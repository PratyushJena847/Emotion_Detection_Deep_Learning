# Real-time Emotion Detection from Webcam

A complete pipeline for detecting 5 emotions (angry, happy, neutral, sad, surprised) from webcam or IP camera  feed using a CNN model with OpenCV Haar Cascade face detection.

## Dataset

- **5 emotion classes**: angry, happy, neutral, sad, surprised
- **~16,000 images** total (~3,000 per class)
- Images are 100x100 grayscale PNG files
- Already collected and organized in folders: `angry/`, `happy/`, `neutral/`, `sad/`, `surprised/`

## Model Architecture

- **3 Conv2D layers**: 32 (5x5) → 64 (3x3) → 128 (3x3) filters
- **BatchNormalization** after each conv layer
- **MaxPooling2D** (2x2) after each conv layer
- **Flatten** → Dense(256) + Dropout(0.5) → Dense(128) + Dropout(0.3)
- **Output**: Dense(5) with Softmax
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical Crossentropy

## Files

| File | Purpose |
|------|---------|
| `Consolidate_data.ipynb` | Load images from 5 folders, resize to 100x100 grayscale, save as pickle |
| `emotion_model_training.ipynb` | Train CNN model, save as `emotion_model.keras` |
| `emotion_recognize.ipynb` | Real-time webcam detection with face rectangles & emotion labels |

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run consolidation (creates `data/images.p`, `data/labels.p`)
Open and run all cells in `Consolidate_data.ipynb`

### 3. Train the model (creates `emotion_model.keras`)
Open and run all cells in `emotion_model_training.ipynb`

### 4. Run real-time detection
Open and run all cells in `emotion_recognize.ipynb`

Press **'q'** to quit the webcam window.

## Output

- **Colored rectangle** around detected face
- **Emotion label** with confidence percentage above rectangle
- Color coding: 🔴 Angry | 🟢 Happy | 🟡 Neutral | 🔵 Sad | 🟠 Surprised

## Notes

- Model file (`emotion_model.keras`) and dataset folders are gitignored (large files)
- Webcam detection falls back to IP camera (http://10.192.35.82:8080/shot.jpg) if local webcam unavailable
- On Windows, TensorFlow GPU is not supported natively (uses CPU)