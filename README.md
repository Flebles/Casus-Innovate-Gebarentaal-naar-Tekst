# Webcam Hand Tracker: Sign Language Recognition Demonstrator
## Gebarentaal naar Tekst (Sign Language to Text)

A real-time hand gesture recognition application using OpenCV and MediaPipe to detect and track hand landmarks in real-time. Built as a demonstrator for Zuyd Hogeschool's Lectorate Data Intelligence to showcase AI applications in gesture and sign language recognition.

### Project Information

**Institution:** Zuyd Hogeschool - Lectorate Data Intelligence  
**Case Group:** TiMMU  
**Academic Year:** 2025-2026

**Team Members:**
- Roy Teheux (2506589)
- Bram Noortman (2507422)
- Noah Siemers (2505774)
- Tim Smeets (2506878)
- Milan Schoenmakers (2502547)

**Purpose:** Create an accessible, beginner-friendly demonstrator that showcases how AI and computer vision can be applied to recognize and interpret gesture and sign language. Designed for open days and public events at Zuyd Hogeschool.

## Features

- 🎥 Real-time webcam hand tracking
- 🖐️ Detects up to 2 hands simultaneously (configurable to 1)
- 📍 Draws 21 hand landmarks and connections per hand
- 🔄 Smooth real-time processing (~30 FPS)
- 📸 Capture snapshots with `s` key
- ⚙️ Highly configurable detection parameters
- 🎯 Optimized for CPU and GPU processing
- ❌ Clean exit with `q` key

## Requirements

**Minimum:**
- Python 3.10+
- Windows 11 (or Linux/macOS)
- 4GB RAM
- Webcam (minimum 640x480 resolution)

**Recommended:**
- Python 3.13+
- 8GB+ RAM
- GPU (CUDA-compatible for acceleration)
- Webcam with 30+ FPS

## Installation

### 1. Clone or download the repository
```powershell
cd "C:\Users\[YourUsername]\Documents"
git clone [repository-url]
cd Casus-Innovate-Gebarentaal-naar-Tekst
```

### 2. Create virtual environment
```powershell
python -m venv .venv
```

### 3. Activate virtual environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This will automatically download the MediaPipe hand landmarker model on first run.

## Running the Application

### Basic usage
```powershell
python CameraControls.py
```

### With custom options
```powershell
python CameraControls.py --camera-index 0 --max-num-hands 2 --min-detection 0.6 --min-tracking 0.5 --snapshot-dir snapshots
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--camera-index` | 0 | Index of the camera to use (0 = default) |
| `--max-num-hands` | 2 | Maximum number of hands to detect (1-2) |
| `--min-detection` | 0.5 | Minimum detection confidence (0.0-1.0) |
| `--min-tracking` | 0.5 | Minimum tracking confidence (0.0-1.0) |
| `--snapshot-dir` | snapshots | Directory to save snapshots |

## Keyboard Controls

| Key | Action |
|-----|--------|
| `s` | Save snapshot of current frame |
| `q` | Quit application |

## Quick Check

```powershell
python CameraControls.py --help
```

## Testing & Documentation

Detailed test results and system validation can be found in:
- **TEST_RAPPORT.txt** - Comprehensive test report with 10 test cases (100% passing)
  - Installation validation
  - Functionality testing
  - Performance benchmarks
  - System stability checks
  - Troubleshooting guide

## Troubleshooting

### No webcam window appears
- Check Windows camera permissions (Settings > Privacy > Camera)
- Ensure no other application is using the camera
- Try a different camera index: `--camera-index 1` or `--camera-index 2`

### Camera not detected as default
Try specifying a different camera index:
```powershell
python CameraControls.py --camera-index 1
```

### Low detection accuracy
- Improve lighting conditions
- Use a clean, uncluttered background
- Adjust detection thresholds:
  ```powershell
  python CameraControls.py --min-detection 0.7 --min-tracking 0.6
  ```

### Performance issues
- Reduce hand tracking load: `--max-num-hands 1`
- Lower detection sensitivity if needed
- Consider using GPU acceleration

## Technical Stack

- **Python 3.13** - Programming language
- **OpenCV 4.8.1.78** - Computer vision library for video capture and processing
- **MediaPipe 0.10.33** - Hand detection and landmark tracking
- **NumPy** - Numerical computations
- **TensorFlow Lite** - Optimized inference for real-time processing

## Project Background

This project is part of the "Casus Innovate" initiative at Zuyd Hogeschool's Data Intelligence Lectorate. The goal is to create an accessible demonstrator that:
- Shows how AI and computer vision can be applied to gesture recognition
- Makes sign language accessibility technology understandable to the public
- Demonstrates both capabilities and limitations of current AI technology
- Serves as an educational tool for open days and public events
- Focuses on Dutch Sign Language (Nederlandse Gebarentaal - NGT)

### Research Foundation

The project is built on extensive research into sign language recognition, computer vision, and AI/ML frameworks. Key concepts include:
- **Pose Estimation** for extracting hand keypoints (21 landmarks per hand)
- **MediaPipe Solutions** for real-time hand tracking
- **Real-time Processing** for live video analysis with minimal latency
- **Accessibility-first Design** for inclusive technology demonstration

## Notes

- This is a demonstration application designed for controlled environments
- Performance may vary based on lighting, background, and camera quality
- The application works best with stable lighting and uncluttered backgrounds
- Maximum 2 hands can be detected simultaneously

## Future Enhancements

**Phase 2 - Gesture Recognition:**
- Gesture classification and pattern recognition
- Database of recognized sign language gestures
- Real-time text output with confidence scores

**Phase 3 - Advanced Features:**
- Recording and playback functionality
- Multi-language sign language support
- Performance optimization with GPU acceleration
- Explainability features for educational purposes

## License

Educational project for Zuyd Hogeschool

## Contact

For questions or issues, contact the TiMMU case group team.
