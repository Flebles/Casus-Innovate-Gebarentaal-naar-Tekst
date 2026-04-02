# Hand Tracker (MediaPipe + OpenCV)

A real-time hand tracking app using your webcam.

## What it does

- Detects up to 2 hands (configurable)
- Draws hand landmarks and connections
- Shows left/right handedness labels
- Shows current FPS
- Saves a snapshot when you press `s`

## Setup (Windows PowerShell)

```powershell
cd "C:\Users\noort\OneDrive\Documenten\GitHub\Casus-Innovate-Gebarentaal-naar-Tekst"
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python .\CameraControl.py
```

## Optional arguments

```powershell
python .\CameraControl.py --camera-index 0 --max-num-hands 2 --min-detection 0.6 --min-tracking 0.5 --snapshot-dir snapshots
```

## Controls

- `s`: Save snapshot
- `q`: Quit

## Notes

- If no window appears, check camera permissions in Windows privacy settings.
- If your camera is not the default one, try another `--camera-index` like `1` or `2`.

