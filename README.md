
# 🎙️ Voice of Truth — Audio Deepfake Detection

> Detecting audio deepfakes using spectrograms, CNN, and SHAP explainability.


## 📌 Overview
Audio deepfakes pose a serious threat in misinformation, fraud, and digital security.  
This project implements a **deep learning–based system** to detect whether a given audio clip is **real or fake**.  

✅ Converts audio into spectrograms  
✅ Trains a **CNN classifier**  
✅ Uses **SHAP** to explain predictions  
✅ Provides a **Streamlit demo app** for testing audio clips  

---

## 🏗️ Architecture

![Architecture](assets/architecture.png)  
*(Example: Audio → Spectrogram → CNN → Classification + SHAP)*

1. **Preprocessing** – Convert `.wav` into mel-spectrograms.  
2. **Modeling** – Train a CNN to classify *real* vs *fake*.  
3. **Explainability** – Apply SHAP for local + global interpretation.  
4. **Deployment** – Streamlit app to upload audio and see results live.  

---

## 📊 Example Results

| Input Audio | Prediction | Confidence | SHAP Explanation |
|-------------|------------|------------|------------------|
| `real_01.wav` | ✅ Real | 0.92 | ![shap](assets/shap_real.png) |
| `fake_05.wav` | ❌ Fake | 0.88 | ![shap](assets/shap_fake.png) |

🔬 Technologies Used

Python 3.10
Librosa – audio processing

TensorFlow / PyTorch – deep learning models

SHAP – explainability

Streamlit – web demo

Scikit-learn – metrics, preprocessing

📈 Performance

Dataset: SceneFake / FoR (Kaggle)

Best model: CNN with spectrogram input

Accuracy: ~91% on test set

SHAP explanations highlight frequency-time regions influencing predictions


