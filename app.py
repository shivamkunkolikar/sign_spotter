import os, io, base64, threading, time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO

# ── CNN model (kept for photo-classification mode) ────────
class CNN_model(nn.Module):
    def __init__(self, num_classes):
        super(CNN_model, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5), 
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=5), 
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        return self.fc_layers(x)

# ── Config ─────────────────────────────────────────────────────
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CNN_MODEL_PATH  = "best_model26_best.pth"
YOLO_MODEL_PATH = "best.pt"          # ← your trained GTSDB YOLOv8 weights

YOLO_CONF       = 0.50               # detection confidence threshold
YOLO_IOU        = 0.45               # NMS IoU threshold
CNN_THRESHOLD   = 0.85               # for photo-classify mode only
MIN_SIZE        = 15                 # minimum box side in pixels

CLASS_NAMES = [
    'Speed limit (20km/h)','Speed limit (30km/h)','Speed limit (50km/h)',
    'Speed limit (60km/h)','Speed limit (70km/h)','Speed limit (80km/h)',
    'End of speed limit (80km/h)','Speed limit (100km/h)','Speed limit (120km/h)',
    'No passing','No passing veh over 3.5t','Right-of-way at intersection',
    'Priority road','Yield','Stop','No vehicles','Veh > 3.5t prohibited',
    'No entry','General caution','Dangerous curve left','Dangerous curve right',
    'Double curve','Bumpy road','Slippery road','Road narrows on the right',
    'Road work','Traffic signals','Pedestrians','Children crossing',
    'Bicycles crossing','Beware of ice/snow','Wild animals crossing',
    'End speed + passing limits','Turn right ahead','Turn left ahead',
    'Ahead only','Go straight or right','Go straight or left',
    'Keep right','Keep left','Roundabout mandatory',
    'End of no passing','End no passing veh over 3.5t'
]

# ── Load models ────────────────────────────────────────────────
print("[INFO] Loading CNN (for photo-classify)...")
cnn_model = CNN_model(num_classes=43)
checkpoint = torch.load(CNN_MODEL_PATH, map_location=DEVICE)
cnn_model.load_state_dict(checkpoint['model_state_dict'])
cnn_model.to(DEVICE).eval()

print("[INFO] Loading trained GTSDB YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print(f"[INFO] Models ready — running on {DEVICE}")

# ── CNN transform (photo-classify mode ) ───────────────────
transform_classify = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(mean=(0.3403, 0.3121, 0.3214),
                std =(0.2724, 0.2608, 0.2669))
])


# ── Photo classification ( uses CNN) ──────────
def classify_image(pil_img):
    inp = transform_classify(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = cnn_model(inp)
        probs  = torch.softmax(logits, dim=1)[0]
    top5_conf, top5_idx = probs.topk(5)
    return [
        {"class": CLASS_NAMES[i.item()], "confidence": round(c.item() * 100, 1)}
        for c, i in zip(top5_conf, top5_idx)
    ]


# ── Detection using trained GTSDB YOLO ─
def detect_frame(frame_bgr):
    
    results = yolo_model.predict(
        source=frame_bgr,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        verbose=False,
        device=DEVICE,
    )[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1

        # Skip tiny boxes
        if w < MIN_SIZE or h < MIN_SIZE:
            continue

        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])

        # Guard against index out of range
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"

        detections.append({
            "box":        [x1, y1, x2, y2],
            "label":      label,
            "confidence": round(conf * 100, 1),
        })

    return detections


# ── Drawing helper (unchanged) ─────────────────────────────────
def draw_detections(frame_bgr, detections):
    out = frame_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f'{d["label"]} {d["confidence"]}%'
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 90), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 150, 60), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ── Webcam stream ──────────────────────────────────────────────
class WebcamStream:
    def __init__(self):
        self.cap     = None
        self.running = False
        self.lock    = threading.Lock()
        self.latest  = None

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            dets = detect_frame(frame)
            ann  = draw_detections(frame, dets)
            _, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self.lock:
                self.latest = buf.tobytes()
            time.sleep(0.03)

    def get_frame(self):
        with self.lock:
            return self.latest


webcam = WebcamStream()


# ── Video file stream ──────────────────────────────────────────
class VideoStream:
    def __init__(self):
        self.cap          = None
        self.running      = False
        self.paused       = False
        self.lock         = threading.Lock()
        self.latest_jpg   = None
        self.latest_dets  = []
        self.progress     = 0.0
        self.total_frames = 1
        self.seen_signs   = set()
        self.fps          = 30.0
        self.path         = None

    def load(self, path: str):
        self.path = path
        cap = cv2.VideoCapture(path)
        self.total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

    def start(self):
        if self.running:
            return
        self.cap      = cv2.VideoCapture(self.path)
        self.running  = True
        self.paused   = False
        self.progress = 0.0
        self.seen_signs.clear()
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.paused  = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def toggle_pause(self):
        self.paused = not self.paused

    def seek(self, pct: float):
        if self.cap:
            target = int((pct / 100.0) * self.total_frames)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)

    def _loop(self):
        frame_delay = 1.0 / self.fps
        frame_idx   = 0

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            t0 = time.time()
            ret, frame = self.cap.read()

            if not ret:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            dets = detect_frame(frame)
            for d in dets:
                self.seen_signs.add(d["label"])

            ann = draw_detections(frame, dets)

            # Progress bar overlay
            h, w   = ann.shape[:2]
            prog_px = int((frame_idx / self.total_frames) * w)
            cv2.rectangle(ann, (0, h - 4), (w, h),       (30, 30, 30),   -1)
            cv2.rectangle(ann, (0, h - 4), (prog_px, h), (0, 220, 90),   -1)

            _, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 82])
            with self.lock:
                self.latest_jpg  = buf.tobytes()
                self.latest_dets = dets
                self.progress    = round(frame_idx / self.total_frames * 100, 1)

            frame_idx += 1

            elapsed = time.time() - t0
            sleep   = frame_delay - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def get_frame(self):
        with self.lock:
            return self.latest_jpg

    def get_state(self):
        with self.lock:
            return {
                "running":    self.running,
                "paused":     self.paused,
                "progress":   self.progress,
                "detections": self.latest_dets,
                "seen_signs": sorted(self.seen_signs),
                "fps":        round(self.fps, 1),
            }


video_stream = VideoStream()


# ── Flask app ──────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", class_names=CLASS_NAMES)

# Mode 3 — photo classification (CNN, unchanged)
@app.route("/api/classify", methods=["POST"])
def api_classify():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "No image"}), 400
    pil = Image.open(f.stream)
    if pil.mode in ("RGBA", "P"):
        pil = pil.convert("RGB")
    top5 = classify_image(pil)
    pil.thumbnail((300, 300))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    thumb_b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({"predictions": top5, "thumbnail": thumb_b64})

# Mode 2 — video file
@app.route("/api/video/upload", methods=["POST"])
def api_video_upload():
    f = request.files.get("video")
    if not f:
        return jsonify({"error": "No video"}), 400
    tmp = "/tmp/uploaded_video.mp4"
    f.save(tmp)
    video_stream.stop()
    video_stream.load(tmp)
    return jsonify({"status": "ready",
                    "total_frames": video_stream.total_frames,
                    "fps": video_stream.fps})

@app.route("/api/video/start", methods=["POST"])
def api_video_start():
    video_stream.start()
    return jsonify({"status": "started"})

@app.route("/api/video/stop", methods=["POST"])
def api_video_stop():
    video_stream.stop()
    return jsonify({"status": "stopped"})

@app.route("/api/video/pause", methods=["POST"])
def api_video_pause():
    video_stream.toggle_pause()
    return jsonify({"paused": video_stream.paused})

@app.route("/api/video/seek", methods=["POST"])
def api_video_seek():
    pct = float(request.json.get("pct", 0))
    video_stream.seek(pct)
    return jsonify({"status": "ok"})

@app.route("/api/video/state")
def api_video_state():
    return jsonify(video_stream.get_state())

@app.route("/api/video/feed")
def api_video_feed():
    def generate():
        while video_stream.running:
            frame = video_stream.get_frame()
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + frame + b"\r\n")
            time.sleep(0.03)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# Mode 1 — webcam
@app.route("/api/webcam/start", methods=["POST"])
def webcam_start():
    try:
        webcam.start()
        return jsonify({"status": "started"})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/webcam/stop", methods=["POST"])
def webcam_stop():
    webcam.stop()
    return jsonify({"status": "stopped"})

@app.route("/api/webcam/snapshot")
def webcam_snapshot():
    frame = webcam.get_frame()
    if frame is None:
        return jsonify({"detections": []})
    arr  = np.frombuffer(frame, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    dets = detect_frame(img)
    return jsonify({"detections": dets})

@app.route("/api/webcam/feed")
def webcam_feed():
    def generate():
        while webcam.running:
            frame = webcam.get_frame()
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + frame + b"\r\n")
            time.sleep(0.03)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)