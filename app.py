#!/usr/bin/env python3
"""
app.py - Flask wrapper for deepfake detector with embedded DetectorModel.

Drop this file in your project root (next to outputs/, templates/, uploads/).
Run: python app.py
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
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ------------------ Config ------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOADS = BASE_DIR / "uploads"
UPLOADS.mkdir(exist_ok=True)
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

MODEL_PATHS = [
    str(BASE_DIR / "outputs" / "best_model.pth"),
    str(BASE_DIR / "outputs" / "final_model.pth"),
]
CALIB_PATH = str(BASE_DIR / "outputs" / "calibrator.joblib")
DEFAULT_IMG_SIZE = 128
DEFAULT_MODEL_NAME = "efficientnet_b0"

# ------------------ DetectorModel (copied from training file) ------------------
import timm  # ensure timm is installed in your venv

class DetectorModel(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = True, drop_rate: float = 0.3, img_size: int = 224):
        super().__init__()
        # create backbone with no classifier
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            # probe with dummy input on cpu to infer feature dims
            self.backbone.eval()
            with torch.no_grad():
                dummy = torch.zeros(1,3,img_size,img_size)
                feats = self.backbone.forward_features(dummy)
                if feats.dim() == 4:
                    feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1).view(1, -1)
                feat_dim = feats.shape[1]
        self.feat_dim = int(feat_dim)
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate/2),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 4:
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1).view(feats.size(0), -1)
        logits = self.head(feats).squeeze(1)
        return logits

# ------------------ Flask app ------------------
app = Flask(__name__)
app.secret_key = "replace-with-a-secure-random-string"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT

# --------------- Model loading & preprocess helpers ---------------
def find_model_path():
    for p in MODEL_PATHS:
        if Path(p).exists():
            return p
    return None

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_model_ckpt(model_path: str, model_name: str = DEFAULT_MODEL_NAME, img_size: int = DEFAULT_IMG_SIZE):
    device = get_device()
    ckpt = torch.load(model_path, map_location=device)
    # instantiate model and load state
    model = DetectorModel(backbone_name=model_name, pretrained=False, img_size=img_size)
    # checkpoint can be dict with model_state_dict or raw state dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)
    model = model.to(device).eval()
    ckpt_args = {}
    if isinstance(ckpt, dict) and "args" in ckpt:
        ckpt_args = ckpt["args"]
    return model, ckpt_args, device

def load_calibrator(path: str):
    if not Path(path).exists():
        return None
    try:
        iso = joblib.load(path)
        if not hasattr(iso, "predict"):
            return None
        return iso
    except Exception:
        return None

def center_crop_and_resize_pil(img: Image.Image, size: int) -> Image.Image:
    w,h = img.size
    m = min(w,h)
    left = (w - m)//2
    top = (h - m)//2
    imc = img.crop((left, top, left + m, top + m)).resize((size, size), Image.LANCZOS)
    return imc

def preprocess_image(image_path: str, img_size: int = DEFAULT_IMG_SIZE, center_crop: bool = False):
    img = Image.open(image_path).convert("RGB")
    if center_crop:
        img = center_crop_and_resize_pil(img, img_size)
        arr = np.array(img)
        transform = A.Compose([A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()])
        t = transform(image=arr)["image"].unsqueeze(0)
    else:
        transform = A.Compose([A.Resize(img_size, img_size), A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()])
        arr = np.array(img)
        t = transform(image=arr)["image"].unsqueeze(0)
    return t

def predict_with_debug(model, device, image_path: str, calibrator=None, img_size: int = DEFAULT_IMG_SIZE, center_crop: bool = False):
    t = preprocess_image(image_path, img_size=img_size, center_crop=center_crop).to(device)
    with torch.no_grad():
        logits = model(t)
        # ensure scalar
        if isinstance(logits, torch.Tensor):
            logit = float(logits.cpu().numpy().ravel()[0])
        else:
            logit = float(logits)
        raw_prob = float(torch.sigmoid(torch.tensor(logit)).numpy().ravel()[0])
    calibrated = None
    percent = None
    if calibrator is not None:
        try:
            calibrated = float(calibrator.predict([raw_prob])[0])
            percent = float(calibrated * 100.0)
        except Exception:
            calibrated = None
            percent = None
    return {"logit": logit, "raw_prob": raw_prob, "calibrated": calibrated, "percent": percent}

# ------------- Global cache for model & calibrator -------------
_global = {"model": None, "device": None, "ckpt_args": None, "calibrator": None, "model_path": None, "img_size": DEFAULT_IMG_SIZE, "model_name": DEFAULT_MODEL_NAME}

def ensure_model_loaded(model_path: Optional[str] = None, calib_path: Optional[str] = None, model_name: str = DEFAULT_MODEL_NAME, img_size: int = DEFAULT_IMG_SIZE):
    # find model if not provided
    if model_path is None:
        model_path = find_model_path()
    if model_path is None:
        raise FileNotFoundError("No model found in outputs/ (expected best_model.pth or final_model.pth)")
    # reload if path changed or not loaded
    if _global["model"] is None or _global["model_path"] != str(model_path) or _global["img_size"] != img_size or _global["model_name"] != model_name:
        model, ckpt_args, device = load_model_ckpt(model_path, model_name=model_name, img_size=img_size)
        _global["model"] = model
        _global["device"] = device
        _global["ckpt_args"] = ckpt_args
        _global["model_path"] = str(model_path)
        _global["img_size"] = img_size
        _global["model_name"] = model_name
    # calibrator optional
    if _global["calibrator"] is None and calib_path is not None and Path(calib_path).exists():
        _global["calibrator"] = load_calibrator(calib_path)
    return _global["model"], _global["device"], _global["ckpt_args"], _global["calibrator"]

# ------------------ Flask routes ------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("home.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No file part", "danger")
        return redirect(url_for("home"))
    file = request.files["image"]
    if file.filename == "":
        flash("No selected file", "danger")
        return redirect(url_for("home"))
    if not allowed_file(file.filename):
        flash("Unsupported file type", "danger")
        return redirect(url_for("home"))

    use_calib = request.form.get("use_calib", "on") != "off"
    center_crop = request.form.get("center_crop", "off") == "on"

    filename = secure_filename(file.filename)
    save_path = UPLOADS / filename
    file.save(save_path)

    # load model & calibrator
    model_path = find_model_path()
    try:
        model, device, ckpt_args, calibrator = ensure_model_loaded(model_path=model_path, calib_path=CALIB_PATH, model_name=_global["model_name"], img_size=_global["img_size"])
    except Exception as e:
        flash(f"Model load failed: {e}", "danger")
        return redirect(url_for("home"))

    calib_to_use = calibrator if use_calib and calibrator is not None else None

    try:
        debug = predict_with_debug(model, device, str(save_path), calibrator=calib_to_use, img_size=_global["img_size"], center_crop=center_crop)
        result = {
            "filename": filename,
            "logit": debug["logit"],
            "raw_prob": debug["raw_prob"],
            "calibrated": debug["calibrated"],
            "percent": debug["percent"],
            "used_calibrator": calib_to_use is not None,
            "ckpt_args": ckpt_args or {},
            "center_crop": center_crop,
        }
        return render_template("home.html", result=result)
    except Exception as e:
        flash(f"Prediction failed: {e}", "danger")
        return redirect(url_for("home"))

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOADS), filename)

@app.route("/debug", methods=["GET"])
def debug_json():
    """
    Quick JSON endpoint:
    /debug?image=uploads/foo.jpg&use_calib=0&center_crop=1
    """
    image = request.args.get("image")
    use_calib = request.args.get("use_calib", "1") != "0"
    center_crop = request.args.get("center_crop", "0") == "1"
    if not image:
        return jsonify({"ok": False, "error": "no image specified (use ?image=uploads/foo.jpg)"}), 400
    path = Path(image)
    if not path.is_absolute():
        path = BASE_DIR / image
    if not path.exists():
        return jsonify({"ok": False, "error": f"image not found: {path}"}), 404
    try:
        model_path = find_model_path()
        model, device, ckpt_args, calibrator = ensure_model_loaded(model_path=model_path, calib_path=CALIB_PATH, model_name=_global["model_name"], img_size=_global["img_size"])
    except Exception as e:
        return jsonify({"ok": False, "error": f"model load failed: {e}"}), 500
    calib_to_use = calibrator if use_calib and calibrator is not None else None
    try:
        debug = predict_with_debug(model, device, str(path), calibrator=calib_to_use, img_size=_global["img_size"], center_crop=center_crop)
        return jsonify({"ok": True, "image": str(path), "result": debug, "used_calibrator": calib_to_use is not None, "ckpt_args": ckpt_args})
    except Exception as e:
        return jsonify({"ok": False, "error": f"predict failed: {e}"}), 500

if __name__ == "__main__":
    # start-up checks
    model_path = find_model_path()
    if model_path is None:
        print("Warning: no model found in outputs/ (expected best_model.pth or final_model.pth). The app will still start but predictions will fail until a model is present.")
    if not Path(CALIB_PATH).exists():
        print("Warning: calibrator not found at", CALIB_PATH, "(calibration will be disabled until a calibrator exists)")
    print("Starting Flask on 0.0.0.0:5080 (debug True). Model path:", model_path)
    app.run(host="0.0.0.0", port=5080, debug=True)
