import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from fpdf import FPDF

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load CSS
load_css(r"G:\final Audiodeepfake\style_all.css")

# Load model
model = tf.keras.models.load_model("G:/final Audiodeepfake/cnn_model_smaller.keras")

st.title("🔍 Audio Deepfake Detection (Forensic Tool)")
st.write("Upload an audio file (wav/mp3) and the system will analyze it for authenticity.")

# Upload audio
audio_file = st.file_uploader("Upload audio file (wav/mp3)", type=["wav", "mp3"])

if audio_file is not None:
    st.audio(audio_file)

    with st.spinner("Processing audio and generating spectrogram..."):
        # Load audio
        y, sr = librosa.load(audio_file, sr=None, duration=5)

        # Generate spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot and save spectrogram
        fig, ax = plt.subplots()
        librosa.display.specshow(S_dB, sr=sr, ax=ax, cmap='magma')
        plt.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig)

        st.image(buf, caption="Generated Spectrogram", use_column_width=True)

        # Prepare image for model
        img = Image.open(buf).convert('RGB').resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array)
        st.write(f"Raw model output: {pred}")  #  See raw output for debug

        # Determine logic based on output shape
        if pred.shape[1] == 1:
            # Sigmoid case
            prob_fake = pred[0][0]
            label = 'FAKE' if prob_fake > 0.5 else 'REAL'
            confidence = prob_fake if label == 'FAKE' else 1 - prob_fake
        else:
            # Softmax case
            class_idx = np.argmax(pred)
            confidence = pred[0][class_idx]
            label = 'REAL' if class_idx == 0 else 'FAKE'

        st.success(f"Prediction: **{label}** (confidence: {confidence:.2f})")

        # Export PDF
        if st.button("Export Result as PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, 'Audio Deepfake Detection Report', ln=True, align='C')
            pdf.ln(10)

            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 10, f'Prediction: {label}', ln=True)
            pdf.cell(0, 10, f'Confidence: {confidence:.2f}', ln=True)

            # Save image
            temp_img = "temp_spec.png"
            with open(temp_img, "wb") as f:
                f.write(buf.getvalue())
            pdf.image(temp_img, w=150)

            # Save and download
            pdf_buf = io.BytesIO()
            pdf.output(pdf_buf)
            pdf_buf.seek(0)

            st.download_button("Download PDF Report", data=pdf_buf, file_name="deepfake_report.pdf", mime="application/pdf")
