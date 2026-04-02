# Webcam Hand Tracker (OpenCV + MediaPipe)

This app starts your webcam immediately and shows real-time hand landmarks.

## Features

- Starts webcam directly (no voice command needed)
- Detects up to 2 hands (configurable)
- Draws hand landmarks and connections
- Shows left/right hand labels
- Saves a snapshot with `s`
- Quits with `q`

## Setup (Windows PowerShell)

```powershell
cd "C:\Users\noort\OneDrive\Documenten\GitHub\Casus-Innovate-Gebarentaal-naar-Tekst"
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
python .\CameraControl.py
```

## Useful options

```powershell
python .\CameraControl.py --camera-index 0 --max-num-hands 2 --min-detection 0.6 --min-tracking 0.5 --snapshot-dir snapshots
```

## Quick check

```powershell
python .\CameraControl.py --help
```

## Notes

- If no webcam window appears, check Windows camera permissions.
- If your camera is not the default one, try `--camera-index 1` or `--camera-index 2`.
