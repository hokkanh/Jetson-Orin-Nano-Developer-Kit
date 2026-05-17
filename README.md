# Real-Time Depth Data Restoration

Developed during the **Defence Hackathon 2026 x Aalto Defence** (Kova Labs challenge). 

This project provides a deterministic, low-latency signal processing pipeline to clean and restore noisy 16-bit depth camera telemetry on Edge devices. By fixing the raw signal before it reaches autonomous navigation algorithms or AI models, we prevent false positives caused by sensor blind spots.

## Architecture
Optimized for the **NVIDIA Jetson Orin Nano**, the codebase is divided into two lightweight modules:

- **`dataReading.py`**: Efficiently parses `.mcap` ROS2 telemetry frame-by-frame using Python generators, ensuring a zero-memory overhead footprint.
- **`main.py`**: The signal processing engine. It isolates blind spots via thresholding, expands them using morphological dilation, and restores the missing data via median-filtered imputation. Finally, it hosts a Flask server for live MJPEG streaming.

## Tech Stack
- **Hardware:** Edge Computing (NVIDIA Jetson Orin Nano)
- **Software:** Python 3.10+, OpenCV, NumPy, Flask, `mcap-ros2-reader`

## How to Run
*(Note: .mcap telemetry files are excluded from this repo for data ownership and security compliance).*

1. Install dependencies: 
   ```bash
   pip install numpy opencv-python flask mcap-ros2-reader
