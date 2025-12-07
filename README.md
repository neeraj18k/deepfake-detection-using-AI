# 🧠 DeepFake Detector

A high-performance **DeepFake Detection System** that identifies AI-generated faces in images using deep learning. Built with **PyTorch** and **Flask**, this system achieves **99.8% AUC** and **98.1% accuracy** on validation data.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Key Features

- **High Accuracy**: 99.8% AUC and 98.1% validation accuracy
- **Real-time Inference**: Web interface for instant predictions
- **Model Calibration**: Isotonic regression for reliable probabilities
- **Fault Injection**: Experimental training modes for robustness testing
- **Multi-device Support**: Automatic GPU/MPS/CPU detection
- **Production Ready**: Clean API and web interface

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Training](#-training)
- [Web Interface](#-web-interface)
- [API Usage](#-api-usage)
- [Model Performance](#-model-performance)
- [Fault Injection Modes](#-fault-injection-modes)
- [Technical Details](#-technical-details)
- [License](#-license)

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM
- NVIDIA GPU (optional, for faster training)

### Step-by-Step Setup

```bash
# Clone the repository
git clone https://github.com/MoteeshA/DeepFake.git
cd DeepFake

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

If `requirements.txt` is not available, install core dependencies manually:

```bash
pip install torch torchvision torchaudio
pip install flask timm albumentations scikit-learn pillow joblib
```

---

## 🎯 Quick Start

### Using Pre-trained Model

1. **Download pre-trained weights** and place them in the `outputs/` folder:
   - `best_model.pth` - Best validation performance
   - `final_model.pth` - Final trained model  
   - `calibrator.joblib` - Calibration model

2. **Launch the web app**:
   ```bash
   python app.py
   ```

3. **Open your browser** to `http://127.0.0.1:5080`

4. **Upload an image** and get instant real/fake prediction!

### Training from Scratch

```bash
# Standard training (8 epochs, 128px images)
python train_fast_faulty.py --dataset_root Dataset --epochs 8 --img_size 128 --output_dir outputs
```

---

## 📁 Project Structure

```
DeepFake/
│
├── app.py                          # Flask web application
├── train_fast_faulty.py            # Training pipeline with fault injection
│
├── outputs/                        # Model artifacts (created after training)
│   ├── best_model.pth              # Best validation model
│   ├── final_model.pth             # Final trained model
│   └── calibrator.joblib           # Calibration model
│
├── Dataset/                        # Training data (organized structure)
│   ├── Train/
│   │   ├── Real/                   # Real face images
│   │   └── Fake/                   # Fake/DeepFake images
│   └── Validation/
│       ├── Real/
│       └── Fake/
│
├── templates/
│   └── home.html                   # Web interface template
├── uploads/                        # User-uploaded images (auto-created)
├── preproc_data/                   # Preprocessed tensors (auto-created)
└── requirements.txt               # Python dependencies
```

---

## 🏋️‍♂️ Training

### Standard Training

```bash
python train_fast_faulty.py \
    --dataset_root Dataset \
    --epochs 8 \
    --batch_size 32 \
    --img_size 128 \
    --lr 0.001 \
    --use_mps \          # Use Apple Silicon GPU if available
    --output_dir outputs
```

### Advanced Training Options

```bash
# With data augmentation and calibration
python train_fast_faulty.py \
    --dataset_root Dataset \
    --epochs 15 \
    --batch_size 64 \
    --img_size 224 \
    --use_amp \          # Automatic Mixed Precision
    --calibrate \        # Enable probability calibration
    --output_dir outputs
```

### Training Progress Output
```
Epoch 1/8 - Train Loss: 0.2154, Val Loss: 0.0981, Val AUC: 0.9954, Val Acc: 0.9654
Epoch 4/8 - Train Loss: 0.0876, Val Loss: 0.0623, Val AUC: 0.9971, Val Acc: 0.9751  
Epoch 8/8 - Train Loss: 0.0452, Val Loss: 0.0418, Val AUC: 0.9980, Val Acc: 0.9810
```

---

## 🌐 Web Interface

### Starting the Server

```bash
python app.py
```

Server starts at: `http://127.0.0.1:5080`

### Web Interface Features

- **Drag & Drop** image upload
- **Real-time preview** of uploaded images
- **Instant predictions** with confidence scores
- **Model information** display
- **Mobile-responsive** design

### Example Prediction Output

```
🎯 Prediction: Fake
📊 Confidence: 98.4%
🤖 Model: EfficientNet-B0
⚖️ Calibration: Enabled
```

---

## 🔌 API Usage

### JSON API Endpoint

```bash
GET /debug?image=uploads/test.jpg&use_calib=1
```

### Example API Call

```bash
curl "http://127.0.0.1:5080/debug?image=uploads/test_image.jpg&use_calib=1"
```

### API Response Format

```json
{
  "ok": true,
  "image": "uploads/test_image.jpg",
  "result": {
    "logit": 2.5178,
    "raw_prob": 0.924,
    "calibrated": 0.938,
    "percent": 93.8,
    "prediction": "Fake"
  },
  "used_calibrator": true,
  "model_device": "cuda:0"
}
```

---

## 📊 Model Performance

### Validation Metrics (8 Epochs)

| Epoch | Validation AUC | Validation Accuracy |
|-------|----------------|---------------------|
| 1     | 0.9954         | 0.9654              |
| 4     | 0.9971         | 0.9751              |
| 8     | 0.9980         | 0.9810              |

### Final Performance
- **AUC**: 0.9980 (99.8%)
- **Accuracy**: 0.9810 (98.1%)
- **Precision**: 0.978
- **Recall**: 0.984

---

## 🔬 Fault Injection Modes

For research and robustness testing, the training pipeline supports various fault injection modes:

| Mode | Command Flag | Description |
|------|--------------|-------------|
| **Label Noise** | `--faulty_mode label_noise` | Randomly flips training labels |
| **Input Noise** | `--faulty_mode input_noise` | Adds Gaussian noise to images |
| **Shuffle Labels** | `--faulty_mode shuffle_labels` | Shuffles labels within batches |
| **Wrong Loss** | `--faulty_mode wrong_loss` | Uses incorrect loss function |
| **Zero Gradients** | `--zero_grad_every_n N` | Skips gradient updates every N batches |
| **Weight Reset** | `--faulty_mode random_weight_reset` | Randomly resets model weights |

### Example Faulty Training
```bash
# Train with 30% label noise
python train_fast_faulty.py --faulty_mode label_noise --label_noise_frac 0.3

# Train with input corruption
python train_fast_faulty.py --faulty_mode input_noise --input_noise_std 0.1
```

---

## 🛠️ Technical Details

### Model Architecture
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Classifier Head**: 2-layer MLP with Dropout (p=0.2)
- **Input Size**: 128×128×3 (configurable to 224×224)
- **Output**: Single logit → Sigmoid probability

### Training Configuration
- **Loss Function**: BCEWithLogitsLoss
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau
- **Batch Size**: 32-64 (depending on GPU memory)
- **Validation**: After each epoch with full validation set

### Data Augmentation
- Random horizontal flip (p=0.5)
- Color jitter (brightness, contrast, saturation)
- Random rotation (±10 degrees)
- Normalization (ImageNet statistics)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch** team for the excellent deep learning framework
- **timm** library for pretrained models
- **Albumentations** for image augmentations
- **Flask** for the web framework

---

## 📧 Contact

**Moteesh Annadanam**
- GitHub: [@MoteeshA](https://github.com/MoteeshA)
- Project Link: [https://github.com/MoteeshA/DeepFake](https://github.com/MoteeshA/DeepFake)

---

## ⭐ Show your support

If you find this project useful, please give it a star on GitHub!
#   d e e p f a k e - d e t e c t i o n - u s i n g - A I  
 #   d e e p f a k e - d e t e c t i o n - u s i n g - A I  
 #   d e e p f a k e - d e t e c t i o n - u s i n g - A I  
 #   d e e p f a k e - d e t e c t i o n - u s i n g - A I  
 