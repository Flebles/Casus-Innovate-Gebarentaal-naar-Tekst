# Gebarentaal naar Tekst - Gebarenherkenning met AI
## Real-time Hand Gesture Recognition Demonstrator

Een real-time toepassing voor handgebaren herkenning met behulp van OpenCV en MediaPipe die hand kernen in real-time detecteert en volgt. Gebouwd als demonstrator voor het Lectorate Data Intelligence van Zuyd Hogeschool om AI-toepassingen in gebarenherkenning en Nederlandse Gebarentaal (NGT) aan te tonen.

### Projectinformatie

**Instelling:** Zuyd Hogeschool - Lectorate Data Intelligence  
**Casus Groep:** TiMMU  
**Academisch jaar:** 2025-2026

**Teamleden:**
- Roy Teheux (2506589)
- Bram Noortman (2507422)
- Noah Siemers (2505774)
- Tim Smeets (2506878)
- Milan Schoenmakers (2502547)

**Doel:** Een toegankelijke demonstrator ontwikkelen die aantoont hoe AI en computer vision kunnen worden toegepast om gebaren en gebarentaal te herkennen en interpreteren. Ontworpen voor open dagen en publieke evenementen bij Zuyd Hogeschool.

## Functies

- Real-time webcam-handtracking
- Detecteert tot 2 handen tegelijk (configureerbaar op 1)
- Toont 21 hand-oriëntatiepunten en verbindingen per hand
- Soepele real-time verwerking (ongeveer 30 FPS)
- Snapshot opnemen met toets 's'
- Zeer configureerbare detectieparameters
- Geoptimaliseerd voor CPU- en GPU-verwerking
- Schoon afsluiten met toets 'q'

## Vereisten

**Minimaal:**
- Python 3.10+
- Windows 11 (of Linux/macOS)
- 4GB RAM
- Webcam (minimaal 640x480 resolutie)

**Aanbevolen:**
- Python 3.13+
- 8GB+ RAM
- GPU (CUDA-compatibel voor versnelling)
- Webcam met 30+ FPS

## Installatie

### 1. Repository klonen of downloaden
```powershell
cd "C:\Users\[JeGebruikersnaam]\Documents"
git clone [repository-url]
cd Casus-Innovate-Gebarentaal-naar-Tekst
```

### 2. Virtuele omgeving aanmaken
```powershell
python -m venv .venv
```

### 3. Virtuele omgeving activeren
```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Afhankelijkheden installeren
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dit zal automatisch het MediaPipe hand landmarker model downloaden bij eerste gebruik.

## De Applicatie Uitvoeren

### Handgebaren herkenning (Live Camera)
```powershell
python main.py
```

### Gebaren Verzamelen (Training data)
```powershell
python main_collect.py --gesture "1" --samples 2000
python main_collect.py --gesture "2" --samples 2000
python main_collect.py --gesture "3" --samples 2000
python main_collect.py --gesture "4" --samples 2000
python main_collect.py --gesture "5" --samples 2000
```

### Model Trainen
```powershell
python main_train.py --dataset data/gestures.csv --model models/ngt_gesture_model.pkl
```

### Real-time Herkenning
```powershell
python main_recognize.py --model models/ngt_gesture_model.pkl --threshold 0.7
```

### Dataset Verifiëren
```powershell
python scripts/verify_dataset.py
```

## Configuratie Opties

| Optie             | Standaard | Beschrijving                               |
|-------------------|-----------|--------------------------------------------|
| `--camera-index`  | 0         | Index van de camera (0 = standaard)        |
| `--max-num-hands` | 2         | Maximaal aantal handen (1-2)               |
| `--min-detection` | 0.5       | Minimale detectiebetrouwbaarheid (0.0-1.0) |
| `--min-tracking`  | 0.5       | Minimale volgbetrouwbaarheid (0.0-1.0)     |
| `--snapshot-dir`  | snapshots | Map voor snapshots                         |

## Toetsenbordbesturing

| Toets | Actie                                         |
|-------|-----------------------------------------------|
| `c`   | Zichtbaarheid van betrouwbaarheid omschakelen |
| `l`   | Markeringslabels omschakelen                  |
| `s`   | Snapshot van huidiggram opslaan               |
| `q`   | Toepassing afsluiten                          |

## Snelle Test

```powershell
python main.py --help
```

## Projectstructuur

```
Casus-Innovate-Gebarentaal-naar-Tekst/
├── src/                          # Bronnodules
│   ├── __init__.py
│   ├── camera.py                 # Handtracking module (HandTracker, CameraController)
│   └── gesture.py                # ML modules (GestureDataManager, GestureModelTrainer, GestureClassifier)
│
├── scripts/                       # Hulpscripts
│   ├── verify_dataset.py         # Controleer verzamelde gegevens kwaliteit
│   └── ngt_demo.py              # NGT-demonstratie (optioneel)
│
├── data/                          # Trainingsgegevens
│   └── gestures.csv              # Geëxtraheerde handmarkeringen
│
├── models/                        # Getrainde modellen
│   └── ngt_gesture_model.pkl     # Getraind RandomForest model
│
├── snapshots/                     # Opgeslagen snapshots
│
├── main.py                        # Hoofdtoepassing (live camera)
├── main_collect.py               # Gebaren verzamelen voor training
├── main_train.py                 # Model trainen
├── main_recognize.py             # Real-time herkenning
├── requirements.txt              # Python-afhankelijkheden
├── README.md                      # Deze file
└── QUICK_START.md               # Snelstartgids
```

## Technische Stack

- **Python 3.13** - Programmeertaal
- **OpenCV 4.8.1.78** - Computer vision bibliotheek voor video verwerking
- **MediaPipe 0.10.33** - Handdetectie en landmark tracking
- **NumPy** - Numerieke berekeningen
- **Scikit-learn** - Machine learning (RandomForest classifier)
- **Pandas** - Dataverwerking
- **TensorFlow Lite** - Geoptimaliseerde gevolgtrekking voor real-time verwerking

## Projectachtergrond

Dit project maakt deel uit van het "Casus Innovate" initiatief van het Lectorate Data Intelligence van Zuyd Hogeschool. Het doel is het creëren van een toegankelijke demonstrator die:
- Aantoont hoe AI en computer vision kunnen worden toegepast op gebarenherkenning
- Gebaren tech voor het publiek begrijpelijk maakt
- Zowel mogelijkheden als beperkingen van huidige AI-technologie demonstreert
- Dient als educatief hulpmiddel voor open dagen en publieke evenementen
- Zich richt op Nederlandse Gebarentaal (NGT)

### Onderzoeksbasis

Het project is gebaseerd op uitgebreid onderzoek naar gebarenherkenning, computer vision en AI/ML-frameworks. Belangrijke concepten:
- **Pose Estimation** voor het extraheren van hand keypoints (21 landmarks per hand)
- **MediaPipe Solutions** voor real-time hand tracking
- **Real-time Processing** voor live video-analyse met minimale latentie
- **Accessibility-first Design** voor inclusieve technologie bemonstering

## Belangrijkste Verbeteringen Gebracht tijdens de ontwikkeling:

### Projectstructuur Gereorganiseerd
- **Nieuwe modulariteit:** Verplaatst code naar `src/` folder (camera.py, gesture.py)
- **Splitsen van verantwoordelijkheden:** Gescheiden tracking van ML in verschillende modules
- **Opgeschoonde imports:** Alle scripts gebruiken nu het nieuwe module systeem

### Gebaaren herkenning Systeem
- **Dataverzameling:** main_collect.py voor het opslaan van hand landmarks
- **Model Training:** main_train.py met RandomForest classifier
- **Real-time Herkenning:** main_recognize.py voor live voorspelling
- **Dataset Validatie:** scripts/verify_dataset.py voor kwaliteitscontrole

### Gegevensverzameling en Training
- Ondersteuning voor meertalige gebaren (nummers 1-5, NGT gebaren)
- Signer-onafhankelijk trainen om het model beter te generaliseren
- MultiSigner ondersteuning met main_collect_multiuser.py (optioneel)

### Code Kwaliteit
- Alle emojis verwijderd uit broncode voor professioneel voorkomen
- Consistent Nederlandse en Engelse commentaar
- Beter gestructureerde error handling

## Probleemoplossing

### Geen webcam window verschijnt
- Controleer Windows camera-machtigingen (Instellingen > Privacy > Camera)
- Zorg dat geen ander programma de camera gebruikt
- Probeer een ander camera-index: `--camera-index 1` of `--camera-index 2`

### Camera niet gedetecteerd als standaard
Probeer een ander camera-index op te geven:
```powershell
python main.py --camera-index 1
```

### Lage detectienauwkeurigheid
- Verbeter de verlichtingscondities
- Gebruik een schone, rommelige achtergrond
- Pas detectiedrempels aan:
  ```powershell
  python main.py --min-detection 0.7 --min-tracking 0.6
  ```

### Prestatieproblemen
- Reduceer hand tracking belasting: `--max-num-hands 1`
- Verlaag detectiegevoeligheid indien nodig
- Overweeg GPU-versnelling te gebruiken

## Opmerkingen

- Dit is een demonstratie toepassing ontworpen voor controleerde omgevingen
- Prestaties kunnen variëren op basis van verlichting, achtergrond en camerakwaliteit
- De toepassing werkt het beste met stabiele verlichting en ongecompliceerde achtergronden
- Maximaal 2 handen kunnen tegelijk worden gedetecteerd

## Toekomstige Verbeteringen

**Fase 2 - Gebarenherkenning:**
- Gebarenclassificatie en patroonherkenning
- Database van erkende gebaren
- Real-time tekstuitvoer met betrouwbaarheidsscores

**Fase 3 - Geavanceerde Functies:**
- Opname- en afspeelfunctionaliteit
- Ondersteuning voor meertalige gebarentaal
- Prestatieoptimalisatie met GPU-versnelling
- Uitlegfuncties voor educatief doeleinden
