#!/usr/bin/env python3
"""
FINAL app.py (UI-COMPATIBLE)
- Image prediction: SAME MODEL
- Fake % & Real % added for UI analytics
- Same `result` object for image & video
- Video prediction logic UNCHANGED
"""

from pathlib import Path
from flask import Flask, request, render_template, redirect, flash, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import uuid
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms

from video_predictor import run_advanced_video_prediction

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent
UPLOADS = BASE_DIR / "uploads"
UPLOADS.mkdir(exist_ok=True)

MODEL_PATHS = [
    BASE_DIR / "outputs" / "best_model.pth",
    BASE_DIR / "outputs" / "final_model.pth"
]

ALLOWED_IMG = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO = {".mp4", ".avi", ".mov"}

DEFAULT_IMG_SIZE = 224
DEFAULT_MODEL_NAME = "efficientnet_b0"

app = Flask(__name__)
app.secret_key = "deepfake-secret"

# ---------------- MODEL ----------------
class DetectorModel(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(drop_rate / 2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1)
        feats = feats.view(feats.size(0), -1)
        return self.head(feats).squeeze(1)

# ---------------- GLOBAL ----------------
_global = {"model": None, "device": None, "img_size": DEFAULT_IMG_SIZE}

def find_model_path():
    for p in MODEL_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("Model file not found")

def ensure_model_loaded():
    if _global["model"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(find_model_path(), map_location=device)

        img_size = ckpt.get("args", {}).get("img_size", DEFAULT_IMG_SIZE)
        backbone = ckpt.get("args", {}).get("backbone_name", DEFAULT_MODEL_NAME)

        model = DetectorModel(backbone_name=backbone)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)

        _global.update({
            "model": model.to(device).eval(),
            "device": device,
            "img_size": img_size
        })

    return _global["model"], _global["device"], _global["img_size"]

# ---------------- IMAGE PREDICTION ----------------
def predict_image(img_path):
    model, device, img_size = ensure_model_loaded()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    img = Image.open(img_path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(t).item()
        raw_prob = torch.sigmoid(torch.tensor(logit)).item()

    # ---- DECISION LOGIC (UNCHANGED) ----
    if raw_prob >= 0.60:
        label = "FAKE (AI-generated)"
    elif raw_prob >= 0.15:
        label = "FAKE (Edited appearance)"
    else:
        label = "REAL"

    fake_percent = round(raw_prob * 100, 2)
    real_percent = round(100 - fake_percent, 2)

    return {
        "raw_prob": round(raw_prob, 4),
        "fake_percent": fake_percent,
        "real_percent": real_percent,
        "label": label
    }

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file or file.filename == "":
        flash("No file uploaded")
        return redirect("/")

    ext = Path(file.filename).suffix.lower()
    filename = secure_filename(file.filename)
    input_path = UPLOADS / filename
    file.save(input_path)

    # ---------- IMAGE ----------
    if ext in ALLOWED_IMG:
        result = predict_image(str(input_path))
        return render_template(
            "home.html",
            filename=filename,
            result=result
        )

    # ---------- VIDEO ----------
    if ext in ALLOWED_VIDEO:
        model, device, img_size = ensure_model_loaded()
        output_name = f"result_{uuid.uuid4().hex}.mp4"
        output_path = UPLOADS / output_name

        video_result = run_advanced_video_prediction(
            str(input_path),
            model,
            device,
            img_size,
            str(output_path)
        )

        # UI-COMPATIBLE RESULT OBJECT
        result = {
            "fake_percent": round(video_result["fake_percent"], 2),
            "real_percent": round(video_result["real_percent"], 2),
            "raw_prob": round(video_result["fake_percent"] / 100, 4),
            "label": "VIDEO ANALYSIS"
        }

        return render_template(
            "home.html",
            video_preview=output_name,
            result=result
        )

    flash("Unsupported file type")
    return redirect("/")

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOADS, filename)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("Running at http://127.0.0.1:5080")
    app.run(host="0.0.0.0", port=5080, debug=True)
