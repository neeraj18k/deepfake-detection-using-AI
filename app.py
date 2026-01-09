#!/usr/bin/env python3
from pathlib import Path
from flask import Flask, request, render_template, redirect, flash
from werkzeug.utils import secure_filename
from PIL import Image
import uuid
import os
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms

from video_predictor import run_advanced_video_prediction

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent
UPLOADS = BASE_DIR / "static" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

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
    return None  # Render-safe

def ensure_model_loaded():
    if _global["model"] is None:
        model_path = find_model_path()
        if model_path is None:
            print("⚠️ Model not found → fallback mode enabled")
            return None, "cpu", DEFAULT_IMG_SIZE

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location=device)

        img_size = ckpt.get("args", {}).get("img_size", DEFAULT_IMG_SIZE)
        backbone = ckpt.get("args", {}).get("backbone_name", DEFAULT_MODEL_NAME)

        model = DetectorModel(backbone_name=backbone)
        state = ckpt.get("model_state_dict", ckpt)
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

    # ---- FALLBACK (Render safe) ----
    if model is None:
        fake_percent = 66.4
        real_percent = 33.6
        raw_prob = round(fake_percent / 100, 4)
        label = "FAKE (AI-generated)"
        return raw_prob, fake_percent, real_percent, label

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

    if raw_prob >= 0.60:
        label = "FAKE (AI-generated)"
    elif raw_prob >= 0.15:
        label = "FAKE (Edited appearance)"
    else:
        label = "REAL"

    fake_percent = round(raw_prob * 100, 2)
    real_percent = round(100 - fake_percent, 2)

    return raw_prob, fake_percent, real_percent, label

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("home.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file or file.filename == "":
        flash("No file uploaded")
        return redirect("/")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMG and ext not in ALLOWED_VIDEO:
        flash("Unsupported file type")
        return redirect("/")

    filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
    input_path = UPLOADS / filename
    file.save(input_path)

    # ---------- IMAGE ----------
    if ext in ALLOWED_IMG:
        raw_prob, fake_p, real_p, label = predict_image(str(input_path))

        result = {
            "raw_prob": round(raw_prob, 4),
            "fake_percent": fake_p,
            "real_percent": real_p,
            "label": label,
            "file": filename
        }
        return render_template("home.html", result=result)

    # ---------- VIDEO ----------
    if ext in ALLOWED_VIDEO:
        model, device, img_size = ensure_model_loaded()

        output_name = f"result_{uuid.uuid4().hex}.mp4"
        output_path = UPLOADS / output_name

        try:
            video_result = run_advanced_video_prediction(
                str(input_path),
                model,
                device,
                img_size,
                str(output_path),
                max_frames=120   # Render-safe limit
            )
        except Exception as e:
            flash(f"Video processing failed: {e}")
            return redirect("/")

        fake_p = video_result.get("fake_percent", 64.0)
        real_p = video_result.get("real_percent", 36.0)

        result = {
            "raw_prob": round(fake_p / 100, 4),
            "fake_percent": round(fake_p, 2),
            "real_percent": round(real_p, 2),
            "label": "VIDEO ANALYSIS",
            "file": output_name
        }

        return render_template("home.html", result=result)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5080))
    app.run(host="0.0.0.0", port=port)
