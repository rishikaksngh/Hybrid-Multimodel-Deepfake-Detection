# Hybrid Multimodal Deepfake Detection System

## About the Project

This project presents a Hybrid Multimodal Deepfake Detection System that combines video-based and audio-based Deep Learning models to identify manipulated media.

Instead of analysing only video frames or audio signals, the system uses both modalities to make a final prediction.

## Project Architecture

The project consists of two independent CNN models:

1. **Video CNN:** Analyses extracted video frames to identify visual manipulation patterns.
2. **Audio CNN:** Uses MFCC features to analyse audio and identify suspicious voice patterns.

The predictions from both models are combined using weighted probability fusion:

`Final Probability = (0.6 × Video Probability) + (0.4 × Audio Probability)`

The final output is classified as REAL or FAKE.

## Technologies Used

- Python
- PyTorch
- OpenCV
- Librosa
- FFmpeg
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Repository Structure

- `notebooks/` – Original project notebook.
- `src/data_pipeline.py` – Dataset preparation and preprocessing.
- `src/video_model.py` – Video CNN architecture and training.
- `src/audio_model.py` – Audio CNN architecture and training.
- `src/fusion_evaluation.py` – Hybrid prediction and evaluation.
- `results/` – Project graphs and evaluation outputs.

## Dataset

The project uses the FakeAVCeleb dataset.

The dataset is not included in this repository. Users must obtain it separately and configure the dataset path before running the project.


## How It Works

1. Load the audio-video dataset.
2. Extract video frames and audio.
3. Preprocess video frames and extract MFCC audio features.
4. Train the video and audio CNN models.
5. Generate predictions from both models.
6. Combine the predictions using weighted fusion.
7. Generate the final REAL or FAKE classification.

## Disclaimer

This is an academic Deep Learning project. Model performance should be interpreted in the context of its dataset, training procedure and independent evaluation.
