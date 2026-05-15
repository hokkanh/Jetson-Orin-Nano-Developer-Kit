# Jetson Orin MCAP Processing Project

## Directory Structure

```
Jetson-Orin-Nano-Developer-Kit/
├── jetson.py                 # Main MCAP processor and system info
├── main.py                   # Entry point (TBD)
├── data/
│   └── mcap_files/          # Place your MCAP video files here
├── output/                   # Processed results (auto-created)
├── models/                   # DL models for object detection (optional)
└── README.md
```

## Quick Start

### 1. Add MCAP Files
Place your MCAP video files in the `data/mcap_files/` directory:

```bash
cp your_video.mcap data/mcap_files/
```

### 2. Install Dependencies

On your local machine:
```bash
pip install mcap mcap-ros2-support opencv-python numpy
```

On Jetson Orin (after SSH connection):
```bash
pip install mcap mcap-ros2-support opencv-python numpy
# For object detection:
pip install ultralytics torch torchvision  # or tensorflow
```

### 3. Run the Processor

Locally (to test):
```bash
python3 jetson.py
```

On Jetson Orin (after pushing files):
```bash
ssh ubuntu@192.168.55.1
cd Jetson-Orin-Nano-Developer-Kit
python3 jetson.py
```

## Transferring Files to Jetson

### Using SCP (SSH Copy)

```bash
# Copy MCAP files to Jetson
scp data/mcap_files/*.mcap ubuntu@192.168.55.1:~/Jetson-Orin-Nano-Developer-Kit/data/mcap_files/

# Copy the entire project
scp -r . ubuntu@192.168.55.1:~/Jetson-Orin-Nano-Developer-Kit/
```

### Using RSYNC (faster for large files)

```bash
rsync -avz --exclude='.git' --exclude='output' data/mcap_files/ ubuntu@192.168.55.1:~/Jetson-Orin-Nano-Developer-Kit/data/mcap_files/
```

## Data Flow

1. **Input**: MCAP files → `data/mcap_files/`
2. **Processing**: `jetson.py` reads and processes files
3. **Output**: Results → `output/`

## Current Status

- ✓ System info gathering
- ✓ MCAP file discovery
- ✓ Dependency checking
- ⏳ MCAP video extraction (next)
- ⏳ Object detection pipeline (next)

## Next Steps

1. Load sample MCAP files to `data/mcap_files/`
2. Implement MCAP frame extraction in `jetson.py`
3. Add YOLOv8/object detection processing
4. Save annotated frames/results to `output/`
