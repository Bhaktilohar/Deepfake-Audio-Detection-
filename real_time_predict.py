import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load your trained model
model = tf.keras.models.load_model(r"G:\final Audiodeepfake\cnn_model_smaller.keras")

# Path to your test spectrogram image (change this!)

# test_img_path = r"G:\final Audiodeepfake\processed_spectrograms\english\real\1.png"
test_img_path = r"G:\final Audiodeepfake\processed_spectrograms\english\real\B_0111_15_D.png"




# Load + preprocess image
img = image.load_img(test_img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prob = model.predict(img_array)[0][0]
label = "FAKE" if prob > 0.5 else "REAL"
confidence = prob if prob > 0.5 else 1 - prob

# Show result
print(f" Prediction: {label} (confidence: {confidence:.2f})")

# Optionally show the image
plt.imshow(img)
plt.title(f"{label} (conf: {confidence:.2f})")
plt.axis('off')
plt.show()
