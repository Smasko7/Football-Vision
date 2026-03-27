# ⚽ FootballVision

A computer vision pipeline for analysing football (soccer) match footage. Given a video of a match, it detects players, tracks them across frames, classifies them into teams, and projects their positions onto a 2D pitch radar view in real time.

<img width="800" alt="Στιγμιότυπο οθόνης (281)" src="https://github.com/user-attachments/assets/d9f32bcc-6779-4952-9f87-3672b04b4c52" />


**What you get:**
- 🟦 Bounding-box and game-style (ellipse/triangle) player annotations
- 🔢 Per-player tracking IDs via ByteTrack
- 👕 Automatic two-team colour separation — no manual labelling required
- 🗺️ 2D minimap / radar view with Voronoi territorial diagrams
- 🎬 Annotated video output for any of the above

---
<img width="800" alt="Στιγμιότυπο οθόνης (282)" src="https://github.com/user-attachments/assets/73ab5360-6daa-46f7-b124-0c4e45ef74fd" />

---

## 📋 Prerequisites

- Python 3.9–3.12 (Python 3.13+ is not supported — the `inference` package requires `<3.13`)
- A [Roboflow](https://roboflow.com) account and API key (free tier works)
- GPU recommended for team-classification modes (falls back to CPU automatically)

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd FootballVision

# 2. Install the sports analytics library (bundled in the repo)
pip install -e sports/

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Setup

### 1. Configure `config.yaml`

Open `config.yaml` and point it at your source video:

```yaml
source_video_path: "videos/your_match.mp4"
```

The output path is **auto-generated** as `outputs/<mode>/<name>_<mode>.mp4` — no need to set it unless you want a custom location.

### 2. Set your Roboflow API key

**Option A — environment variable (recommended):** the key is never committed to version control.

```bash
# Linux / macOS
export ROBOFLOW_API_KEY="YOUR_KEY_HERE"

# Windows PowerShell
$env:ROBOFLOW_API_KEY = "YOUR_KEY_HERE"
```

**Option B — `config.yaml`:**

```yaml
roboflow_api_key: "YOUR_KEY_HERE"
```

> The environment variable always takes precedence over the config file value.

---

## ▶️ Quick Start

Run any mode with:

```bash
python run.py --mode <mode_name>
```

Override paths on the command line without editing the config:

```bash
python run.py --mode radar_video --source match.mp4 --target outputs/radar.mp4
```

### 🗂️ All available modes

| Mode | Command | Output |
|---|---|---|
| 🖼️ Display first frame | `python run.py --mode capture_frame` | Image window |
| 📦 Bounding-box labels | `python run.py --mode basic_annotation` | Image window |
| 🎮 Game-style frame | `python run.py --mode videogame_frame` | Image window |
| 🎮 Game-style video | `python run.py --mode videogame_video` | `outputs/videogame_video/<name>_videogame_video.mp4` |
| 🔢 Player tracking | `python run.py --mode player_tracking` | `outputs/player_tracking/<name>_player_tracking.mp4` |
| 👕 Team clustering | `python run.py --mode team_split` | Image grids (windows) |
| 👕 Team classification video | `python run.py --mode team_video` | `outputs/team_video/<name>_team_video.mp4` |
| 📍 Pitch keypoints | `python run.py --mode pitch_keypoints` | Image window |
| 🗺️ Radar (single frame) | `python run.py --mode radar_frame` | 3 image windows |
| 🗺️ Radar video | `python run.py --mode radar_video` | `outputs/radar_video/<name>_radar_video.mp4` |

---

## 🖥️ Expected Outputs

| Mode | What you see |
|---|---|
| `basic_annotation` | Coloured bounding boxes around every player, goalkeeper, referee, and ball, with class name and confidence score |
| `videogame_frame` / `videogame_video` | Ellipses drawn under players and goalkeepers; a yellow triangle above the ball — similar to a video-game HUD |
| `player_tracking` | Same as above, plus a tracker ID label (`#1`, `#2`, …) that persists across frames |
| `team_split` | Three grids: all player crops, team-0 crops, team-1 crops |
| `team_video` | Players coloured by team (blue = team 0, pink = team 1) |
| `pitch_keypoints` | Frame with pink dots marking detected pitch landmark positions |
| `radar_frame` | (1) Annotated video frame  (2) 2D pitch with coloured player dots  (3) Voronoi diagram showing team territorial control |
| `radar_video` | Video of the 2D pitch radar view with player positions updated each frame |

---

## 🔧 Configuration Reference

All settings live in `config.yaml`. The most commonly changed values are:

| Key | Type | Default | Description |
|---|---|---|---|
| `source_video_path` | string | `videos/Demo_Video.mp4` | Input video file |
| `target_video_path` | string | _(auto)_ | Output path; auto-generated as `outputs/<mode>/<name>_<mode>.mp4` |
| `roboflow_api_key` | string | — | Roboflow API key (prefer env var) |
| `trained_model_path` | string | `""` | Local YOLO model to use instead of Roboflow API _(optional)_ |
| `detection_confidence` | float | `0.3` | Minimum detection confidence |
| `nms_threshold` | float | `0.5` | NMS overlap threshold |
| `annotation_frame_limit` | int | `750` | Frames to process in `videogame_video` (~10 s) |
| `tracking_frame_limit` | int | `250` | Frames to process in `player_tracking` |
| `radar_frame_limit` | int | `250` | Frames to process in `radar_video` |
| `team_video_frame_limit` | int | `10` | Frames to process in `team_video` |
| `embedding_stride` | int | `30` | Frame sampling stride for crop collection |
| `color_team_0` | hex string | `00BFFF` | Team 0 colour (light blue) |
| `color_team_1` | hex string | `FF1493` | Team 1 colour (pink) |

For the full list of settings see `config.yaml`.

---
<img width="800" alt="Στιγμιότυπο οθόνης (283)" src="https://github.com/user-attachments/assets/b75ada9b-ddb5-4f78-945a-710e7385d863" />

---

## 🧠 How It Works

### 1. 🔍 Object Detection
A Roboflow YOLO model (`football-players-detection-3zvbc/15`) detects four classes: **ball**, **goalkeeper**, **player**, **referee**. A second model (`football-field-detection-f07vi/14`) detects pitch landmark keypoints.

### 2. 🔢 Player Tracking
[ByteTrack](https://arxiv.org/abs/2110.06864) (via the `supervision` library) assigns persistent IDs to detections across frames.

### 3. 👕 Team Classification
Player crops are embedded using [SigLIP](https://arxiv.org/abs/2303.15343) (a vision-language model). Embeddings are reduced from 768 to 3 dimensions with UMAP, then split into two clusters with KMeans. This distinguishes the two teams without any manual annotation.

### 4. 🗺️ Pitch Projection
Detected pitch keypoints are matched to known coordinates on a standard 7000 × 12000 cm pitch via a homography matrix (`ViewTransformer`). Player pixel positions are then transformed into pitch coordinates and drawn on a 2D overhead view.

---

## 🏋️ Custom Model Training _(optional)_

You can fine-tune the pretrained YOLO model on your own dataset and use it for inference instead of the Roboflow API.

### 1. Prepare your dataset

Place it in `datasets/` using standard YOLO format:

```
datasets/
└── my_dataset/
    ├── data.yaml          # nc, names, train/val/test paths
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

### 2. Train

```bash
# Use settings from config.yaml
python train.py

# Or override on the command line
python train.py --dataset datasets/my_dataset/data.yaml --epochs 20 --device 0
```

The trained model is exported to `models/` and the path is printed at the end:

```
Model exported to: models/my_dataset_yolo11n.pt
To use this model for inference, set in config.yaml:
  trained_model_path: "models/my_dataset_yolo11n.pt"
```

### 3. Use the trained model

Set `trained_model_path` in `config.yaml`, then run any mode normally:

```bash
python run.py --mode radar_video
```

Leave `trained_model_path` empty to revert to the Roboflow API model.

---

## 📁 Project Structure

```
FootballVision/
├── run.py                 CLI entry point (inference)
├── train.py               CLI entry point (training)
├── config.yaml            All configuration
├── football_vision/       Application package
│   ├── config.py          AppConfig dataclass + config loader
│   ├── core/              Shared utilities (models, detection, annotators, video)
│   └── modes/             One module per analysis mode
├── sports/                Sports analytics library (pitch drawing, TeamClassifier)
├── datasets/              Training datasets (YOLO format)
├── models/                Pretrained and fine-tuned model weights
└── outputs/               Generated videos and embeddings
```


---

## 🙏 Acknowledgements

This project is heavily based on the [roboflow/sports](https://github.com/roboflow/sports) repository by [Roboflow](https://roboflow.com). The core pipeline design, pitch projection logic, team classification approach, and the bundled `sports/` library all originate from that work. This repo extends it with a structured application package, a unified CLI, and a YAML-based configuration system.

---

## ⚠️ Known Limitations

- **GPU strongly recommended** for `team_video`, `radar_frame`, and `radar_video` — SigLIP embedding extraction is slow on CPU.
- **Team assignment is unsupervised** — which cluster maps to which team is arbitrary and may flip between runs.
- **Short clips** affect UMAP + KMeans quality; use a longer clip or lower `embedding_stride` to collect more crops.
- **Pitch projection accuracy** depends on how many keypoints the field model detects with high confidence; poor lighting or unusual camera angles reduce accuracy.
