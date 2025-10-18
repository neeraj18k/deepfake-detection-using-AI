* a full **Flask app (`app.py`)** for inference and UI,
* a **training pipeline (`train_fast_faulty.py`)** with advanced training/fault simulation logic, and
* a trained model (`outputs/best_model.pth` etc.),
  as shown in your screenshot (Epochs 1–8, val_auc ≈ 0.998, val_acc ≈ 0.981).

Here’s a **professional, complete `README.md`** you can directly use for your GitHub repository 👇

---

```markdown
# 🧠 DeepFake Detector

A complete **DeepFake Detection System** built with **PyTorch**, **Flask**, and **EfficientNet**.  
This project detects AI-generated (fake) faces in images and provides calibrated prediction probabilities.

---

## 🚀 Overview

DeepFake Detector is an end-to-end project consisting of:

1. **Model Training Pipeline** — a flexible and experimental trainer with optional *faulty training modes* for robustness studies.
2. **Inference Web App (Flask)** — a simple drag-and-drop image interface that predicts whether an image is *real* or *fake* using a trained model.
3. **Calibration Module** — post-training isotonic regression calibration for more reliable probabilities.

---

## 📁 Project Structure

```

DeepFake-Detector/
│
├── app.py                   # Flask web app for inference
├── train_fast_faulty.py     # Model training script with fault-injection modes
│
├── outputs/                 # Trained model weights and calibrator
│   ├── best_model.pth
│   ├── final_model.pth
│   └── calibrator.joblib
│
├── preproc_data/            # Preprocessed image tensors
├── Dataset/                 # Original training/validation dataset
│   ├── Train/
│   │   ├── Real/
│   │   └── Fake/
│   └── Validation/
│       ├── Real/
│       └── Fake/
│
├── templates/
│   └── home.html            # Frontend UI for the Flask app
├── uploads/                 # Uploaded images for testing
└── requirements.txt

````

---

## ⚙️ Features

✅ **Model**
- Backbone: `EfficientNet-B0` (via `timm`)
- Dropout regularization and 2-layer head  
- Trained with BCEWithLogits loss  
- Optional isotonic calibration  

✅ **Web App**
- Image upload + preview
- Real/fake prediction with calibrated confidence
- JSON debug API endpoint (`/debug`)
- Automatic GPU/MPS/CPU device detection

✅ **Training Options**
- Supports deliberate *faulty training* for research:
  - `label_noise`, `shuffle_labels`, `input_noise`, `wrong_loss`, `shuffle_batch`
  - `zero_grad_every_n`, `random_weight_reset`
- Validation AUC and accuracy per epoch
- Isotonic Regression calibration on validation set

---

## 🧩 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/DeepFake-Detector.git
cd DeepFake-Detector

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # (macOS/Linux)
venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt
````

If `requirements.txt` is not generated yet, create one using:

```bash
pip freeze > requirements.txt
```

---

## 🏋️‍♂️ Training

You can train the detector from scratch or fine-tune existing models.

### Example: Standard Training

```bash
python train_fast_faulty.py --dataset_root Dataset --data_dir data \
    --epochs 8 --batch_size 32 --img_size 128 --use_mps --output_dir outputs
```

### Example: Faulty Training Experiment

```bash
python train_fast_faulty.py --dataset_root Dataset --data_dir data \
    --faulty_mode label_noise --label_noise_frac 0.3 --epochs 8 --output_dir outputs
```

After training, the best weights and calibration model are saved under `outputs/`:

```
outputs/best_model.pth
outputs/final_model.pth
outputs/calibrator.joblib
```

---

## 🧮 Evaluation Example (from training logs)

| Epoch | Val AUC | Val Accuracy |
| ----- | ------- | ------------ |
| 1     | 0.9954  | 0.9654       |
| 4     | 0.9971  | 0.9751       |
| 8     | 0.9980  | 0.9810       |

The final calibrated evaluation reached **98.1% validation accuracy** and **99.8% AUC**.

---

## 🧠 Running the Flask App

Once your model is trained (or copied into `outputs/`), start the web interface:

```bash
python app.py
```

Visit **[http://127.0.0.1:5080](http://127.0.0.1:5080)** or **[http://0.0.0.0:5080](http://0.0.0.0:5080)** in your browser.

---

## 🖼️ Usage

1. Upload an image (`.jpg`, `.jpeg`, `.png`, `.bmp`)
2. Click **Predict**
3. The app displays:

   * Raw probability
   * Calibrated probability (if available)
   * Confidence score (%)

Example result:

```
Prediction: Fake
Confidence: 98.4%
Model: EfficientNet-B0
Calibration: Enabled
```

---

## 🔍 API Debug Endpoint

You can also query predictions directly via API:

```bash
curl "http://127.0.0.1:5080/debug?image=uploads/test.jpg&use_calib=1"
```

Example JSON Response:

```json
{
  "ok": true,
  "image": "uploads/test.jpg",
  "result": {
    "logit": 2.5178,
    "raw_prob": 0.924,
    "calibrated": 0.938,
    "percent": 93.8
  },
  "used_calibrator": true
}
```

---

## 🧩 Tech Stack

* **Language:** Python 3.10+
* **Frameworks:** Flask, PyTorch, Albumentations, timm
* **Visualization:** Bootstrap + Jinja2 Templates
* **Calibration:** scikit-learn Isotonic Regression

---

## 📊 Experimentation (Faulty Training Modes)

| Mode                  | Description                                 |
| --------------------- | ------------------------------------------- |
| `none`                | Normal training                             |
| `label_noise`         | Randomly flips a fraction of labels         |
| `shuffle_labels`      | Shuffles labels within batches              |
| `input_noise`         | Adds Gaussian noise to input images         |
| `wrong_loss`          | Uses MSE instead of BCE (intentional error) |
| `zero_grad_every_n`   | Skips optimizer updates every *n* batches   |
| `random_weight_reset` | Randomly resets a subset of weights         |
| `shuffle_batch`       | Randomizes image-label mapping              |

These modes help in **robustness testing** and **failure analysis** of training pipelines.

---

## 📈 Future Work

* Extend support to video deepfake detection
* Add Grad-CAM visual explanations
* Deploy Flask app as Docker container
* Integrate multi-backbone ensemble (e.g., ViT + EfficientNet)

---

## 🧑‍💻 Author

**Moteesh Annadanam**
Deep Learning & Computer Vision Enthusiast
📍 [GitHub Profile](https://github.com/<your-username>)

---

## 🪪 License

This project is released under the MIT License.
You are free to use, modify, and distribute it with attribution.

---

## 🧠 Acknowledgements

* [PyTorch](https://pytorch.org/)
* [timm](https://github.com/huggingface/pytorch-image-models)
* [Albumentations](https://github.com/albumentations-team/albumentations)
* [Flask](https://flask.palletsprojects.com/)
* [EfficientNet](https://arxiv.org/abs/1905.11946)

```

```
