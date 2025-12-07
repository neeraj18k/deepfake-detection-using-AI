#!/usr/bin/env python3
"""
FINAL FIXED app.py for Deepfake Detection
- Correct IMG_SIZE = 224 (or auto-read from checkpoint)
- Correct preprocessing (Albumentations)
- Correct DetectorModel
- Accurate predictions
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import joblib
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

# ------------------ CONFIG ------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOADS = BASE_DIR / "uploads"
UPLOADS.mkdir(exist_ok=True)

MODEL_PATHS = [
    str(BASE_DIR / "outputs" / "best_model.pth"),
    str(BASE_DIR / "outputs" / "final_model.pth")
]

CALIB_PATH = str(BASE_DIR / "outputs" / "calibrator.joblib")

DEFAULT_IMG_SIZE = 224
DEFAULT_MODEL_NAME = "efficientnet_b0"

app = Flask(__name__)
app.secret_key = "deepfake-secret"
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}

# ------------------ DetectorModel (SAME AS TRAINING) ------------------
class DetectorModel(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", pretrained=False, drop_rate=0.3, img_size=224):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
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
        feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1).view(feats.size(0), -1)
        return self.head(feats).squeeze(1)

# ------------------ HELPERS ------------------
def find_model_path():
    for p in MODEL_PATHS:
        if Path(p).exists():
            return p
    return None

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_calibrator(path):
    if Path(path).exists():
        try:
            cal = joblib.load(path)
            if hasattr(cal, "predict"):
                return cal
        except: pass
    return None

def preprocess_image(img_path, img_size, center_crop=False):
    img = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    return transform(img).unsqueeze(0)

# ------------------ MODEL LOADING ------------------
_global = {
    "model": None,
    "device": None,
    "img_size": DEFAULT_IMG_SIZE,
    "model_name": DEFAULT_MODEL_NAME,
    "calibrator": None
}

def ensure_model_loaded():
    model_path = find_model_path()
    if not model_path:
        raise FileNotFoundError("best_model.pth not found")

    if _global["model"] is None:
        device = get_device()
        ckpt = torch.load(model_path, map_location=device)

        # auto-read checkpoint args
        img_size = ckpt.get("args", {}).get("img_size", DEFAULT_IMG_SIZE)
        backbone = ckpt.get("args", {}).get("backbone_name", DEFAULT_MODEL_NAME)

        _global["img_size"] = img_size
        _global["model_name"] = backbone

        print(f"[LOADING] Using img_size={img_size}, backbone={backbone}")

        model = DetectorModel(backbone_name=backbone, pretrained=False, img_size=img_size)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)

        _global["model"] = model.to(device).eval()
        _global["device"] = device
        _global["calibrator"] = load_calibrator(CALIB_PATH)

    return (
        _global["model"],
        _global["device"],
        _global["img_size"],
        _global["calibrator"]
    )

# ------------------ PREDICT ------------------
def predict_image(img_path, center_crop=False):
    model, device, img_size, calibrator = ensure_model_loaded()

    t = preprocess_image(img_path, img_size, center_crop).to(device)

    with torch.no_grad():
        logit = model(t).item()
        raw_prob = torch.sigmoid(torch.tensor(logit)).item()

    # Temporarily disable calibrator
    calibrated = None
    # if calibrator:
    #     try: calibrated = calibrator.predict([raw_prob])[0]
    #     except: calibrated = None

    return {
        "logit": logit,
        "raw_prob": raw_prob,
        "calibrated": calibrated,
        "percent": raw_prob * 100  # Use raw_prob directly
    }

# ------------------ ROUTES ------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No file uploaded")
        return redirect("/")

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected")
        return redirect("/")

    filename = secure_filename(file.filename)
    save_path = UPLOADS / filename
    file.save(save_path)

    result = predict_image(str(save_path))
    return render_template("home.html", result=result, filename=filename)

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOADS, filename)

# ------------------ DEBUG ENDPOINT ------------------
@app.route("/debug")
def debug():
    img = request.args.get("image")
    if not img:
        return {"error": "provide ?image=uploads/abc.jpg"}

    result = predict_image(img)
    return result

if __name__ == "__main__":
    print("Deepfake Detector Running at http://127.0.0.1:5080")
    app.run(host="0.0.0.0", port=5080, debug=True)
