# 🕵️ Deepfake Face Detection

A deep learning system that detects AI-generated (deepfake) faces with **99.94% confidence**, using a fine-tuned Xception model and Grad-CAM visual explanations.

> Built as a major project — includes both the ML training pipeline and a full Django web app for real-time inference.

---

## Demo

| Real face detected (99.64% confidence) | Fake face detected (99.94% confidence) |
|---|---|
| ![Real](demo/real_result.png) | ![Fake](demo/fake_result.png) |

The **Grad-CAM heatmap** shows exactly which regions of the face the model focused on to make its decision — red/yellow = high attention.

---

## How it works

1. **Face extraction** — faces are cropped from video frames using metadata
2. **Preprocessing** — images resized to 224×224 and normalised
3. **Xception model** — pretrained on ImageNet, fine-tuned on 16,000 real/fake face pairs
4. **Grad-CAM** — generates a heatmap overlay showing what the model "looked at"
5. **Django web app** — upload any face image and get a real-time prediction

---

## Results

| Metric | Score |
|---|---|
| Model | Xception (fine-tuned) |
| Training samples | 16,000 (8,000 real + 8,000 fake) |
| Test accuracy | ~99% |
| Confidence on demo images | 99.64% (real), 99.94% (fake) |

---

## Tech stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=flat&logo=OpenCV&logoColor=white)
![Django](https://img.shields.io/badge/Django-092E20?style=flat&logo=django&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

---

## Project structure

```
deepfake-detection/
├── major_project_code.ipynb   # ML training pipeline (data prep, model, evaluation)
├── xception_deepfake_image.h5 # Trained model weights (download separately)
├── deepfake_app/              # Django web application
│   ├── views.py               # Prediction + Grad-CAM logic
│   ├── templates/
│   │   ├── upload.html        # Upload page
│   │   └── result.html        # Results page with heatmap
│   └── urls.py
├── demo/                      # Demo screenshots
└── requirements.txt
```

---

## Run locally

### 1. Clone the repo

```bash
git clone https://github.com/varsha-sherla/Deepfake-Detection
cd Deepfake-Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the trained model

Place `xception_deepfake_image.h5` in the root directory.  
*(Model trained on the [deepfake-faces](https://www.kaggle.com/datasets/dagnelies/deepfake-faces) dataset from Kaggle)*

### 4. Run the Django app

```bash
python manage.py runserver
```

Open [http://localhost:8000](http://localhost:8000), upload a face image, and get your prediction.

---

## Training the model yourself

Open `major_project_code.ipynb` in Google Colab or Jupyter.

The notebook covers:
- Downloading and exploring the dataset
- Train/validation/test split (stratified)
- Data augmentation (flip, rotation, contrast)
- Fine-tuning Xception with frozen base layers
- Evaluation with confusion matrix and classification report
- Grad-CAM implementation for explainability

---

## Dataset

[Deepfake Faces — Kaggle](https://www.kaggle.com/datasets/dagnelies/deepfake-faces)  
16,000 sampled images (balanced real/fake split).

---

## Author

**Varsha Sherla**  
[LinkedIn](https://www.linkedin.com/in/varsha-s-47103a250) • [GitHub](https://github.com/varsha-sherla)
