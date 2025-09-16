import shap
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load your trained model
model = load_model(r"G:\final Audiodeepfake\cnn_model_smaller.keras")

# Set up data generator (same preprocessing as training)
datagen = ImageDataGenerator(rescale=1./255)
test_gen = datagen.flow_from_directory(
    r"G:\final Audiodeepfake\processed_spectrograms\english",
    target_size=(128, 128),
    batch_size=10,  # small batch for SHAP
    class_mode='binary',
    shuffle=True
)

# Get one batch of data
X_batch, y_batch = next(test_gen)
print(f"Batch shape: {X_batch.shape}")
print(f"True labels: {y_batch}")

# Initialize SHAP
explainer = shap.GradientExplainer(model, X_batch)

# Explain predictions
shap_values = explainer.shap_values(X_batch)

# Plot + save SHAP explanations
for i in range(len(X_batch)):
    shap.image_plot([shap_values], X_batch, show=False)
    plt.savefig(f"shap_explanation_{i}.png")
    plt.close()
    print(f"Saved: shap_explanation_{i}.png")

print(" SHAP explanations saved successfully!")
