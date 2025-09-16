import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import shap
from fpdf import FPDF
from PIL import Image
import io

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load external CSS
load_css(r"G:\final Audiodeepfake\style_all.css")

# Load the model
model = tf.keras.models.load_model(r"G:\final Audiodeepfake\cnn_model_smaller.keras")

st.title(" Audio Deepfake Detection (Spectrogram Input)")
st.write("Upload a spectrogram image (PNG/JPG) and the system will analyze it for authenticity.")

# File uploader for spectrogram image
uploaded_file = st.file_uploader("Upload spectrogram image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Spectrogram", use_column_width=True)

    # Preprocess image
    img = Image.open(uploaded_file).convert('RGB').resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx]

    label = 'REAL' if class_idx == 0 else 'FAKE'
    st.success(f"Prediction: **{label}** (confidence: {confidence:.2f})")

    # Optional SHAP button
    if st.button("Show SHAP Explanation"):
        background = np.zeros((1, 128, 128, 3))
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(img_array)
        
        shap.image_plot(shap_values, img_array)
        st.pyplot(plt.gcf())

    # PDF Export
    if st.button("Export Result as PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, 'Audio Deepfake Detection Report', ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, f'Prediction: {label}', ln=True)
        pdf.cell(0, 10, f'Confidence: {confidence:.2f}', ln=True)

        # Save temp image
        temp_img = "temp_uploaded_spec.png"
        img.save(temp_img)

        pdf.ln(5)
        pdf.image(temp_img, w=150)

        pdf_buf = io.BytesIO()
        pdf.output(pdf_buf)
        pdf_buf.seek(0)

        st.download_button("Download PDF Report", data=pdf_buf, file_name="deepfake_report.pdf", mime="application/pdf")
