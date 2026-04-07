# Webcam Hand Tracker: Sign Language Recognition Demonstrator
## Gebarentaal naar Tekst (Sign Language to Text)

A real-time hand gesture recognition application using OpenCV and MediaPipe to detect and track hand landmarks. Built as a demonstrator for Zuyd Hogeschool's Data Intelligence department to showcase AI applications in sign language translation.

### Project Information

**Case Group:** TiMMU  
**Team Members:**
- Roy Teheux (2506589)
- Bram Noortman (2507422)
- Noah Siemers (2505774)
- Tim Smeets (2506878)
- Milan Schoenmakers (2502547)

**Purpose:** Educational demonstrator for open days and public events to show how AI can be applied to make sign language accessible.

## Features

- 🎥 Starts webcam directly (no voice command needed)
- 🖐️ Detects up to 2 hands (configurable)
- 📍 Draws hand landmarks and connections in real-time
- 🏷️ Shows left/right hand labels
- 📸 Saves a snapshot with `s`
- ❌ Quits with `q`
- ⚙️ Highly configurable detection parameters

## Requirements

- Python 3.8+
- Windows PowerShell (or terminal of your choice)
- Webcam
- 4GB+ RAM recommended
- GPU optional but recommended for better performance

## Setup (Windows PowerShell)

### 1. Navigate to project directory
```powershell
cd "C:\Users\Bram\PycharmProjects\Casus-Innovate-Gebarentaal-naar-Tekst"
```

### 2. Create and activate virtual environment (if not already done)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

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

- **Python 3.8+** - Programming language
- **OpenCV** - Computer vision library
- **MediaPipe** - Hand detection and landmark tracking
- **NumPy** - Numerical computations

## Project Background

This project is part of the "Casus Innovate" initiative at Zuyd Hogeschool's Data Intelligence Lectorate. The goal is to create an accessible demonstrator that:
- Shows how AI and computer vision can be applied to gesture recognition
- Makes sign language accessibility technology understandable to the public
- Demonstrates both capabilities and limitations of current AI technology
- Serves as an educational tool for open days and public events

## Research & References

The project is built on extensive research into sign language recognition, computer vision, and AI/ML frameworks. Key technologies include:
- **Pose Estimation** for extracting hand keypoints
- **Deep Learning Models** for gesture classification
- **Real-time Processing** for live video analysis

For detailed research findings, see the project documentation and bibliography.

## Notes

- This is a demonstration application designed for controlled environments
- Performance may vary based on lighting, background, and camera quality
- The application works best with stable lighting and uncluttered backgrounds
- Maximum 2 hands can be detected simultaneously

## Future Enhancements

- Gesture classification and translation
- Database of recognized sign language gestures
- Real-time text output
- Recording and playback functionality
- Multi-language sign language support

## License

Educational project for Zuyd Hogeschool

## Contact

For questions or issues, contact the TiMMU case group team.
