# sign-spotter 🚦

A fun little web app that detects and classifies German traffic signs in real time using YOLOv8 + a custom CNN — built as a personal experiment, not a serious production thing, may not always give accurate results.

## what it does

- **Webcam mode** — point your camera at a traffic sign and it detects + labels it live
- **Video mode** — upload a dashcam clip and watch it annotate signs frame by frame
- **Photo mode** — upload a single image and get the top 5 classification predictions

## how it works

Two models running together:

- **YOLOv8n** (trained on GTSRB) — finds where the signs are in the frame
- **Custom CNN** — classifies uploaded photos into one of 43 German traffic sign categories

Trained on the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/) — 51,000 images across 43 classes.

## tech stack

- Python, Flask
- PyTorch, Ultralytics YOLOv8
- OpenCV
- Plain HTML/CSS/JS frontend

## setup

```bash
git clone https://github.com/shivamkunkolikar/sign-spotter
cd sign-spotter
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` — model weights are included, no training needed.

## requirements.txt

```
flask
ultralytics
torch
torchvision
opencv-python-headless
pillow
numpy
```

## project structure

```
sign-spotter/
├── app.py
├── best.pt                  ← YOLOv8 detection weights
├── best_model26_best.pth    ← CNN classification weights
├── requirements.txt
├── static/
│   ├── styles.css
│   └── script.js
└── templates/
    └── index.html
```

## notes

- Model was trained on cropped sign images so real-world detection on dashcam footage may vary
- This is just a fun experiment, model may give wrong results

## classes

Detects all 43 GTSRB categories including speed limits, warning signs, priority signs, and prohibitory signs.

---

made for fun ✌️