from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np

# Load your trained model
model = load_model(r"G:\final Audiodeepfake\cnn_model_smaller.keras")

# Path to multilingual spectrograms
lang_dir = r"G:\final Audiodeepfake\processed_spectrograms\lang"

# Generator for lang data
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    lang_dir,
    target_size=(128, 128),
    color_mode='rgb',
    batch_size=1,  # small batch as data is small
    class_mode='binary',
    shuffle=False
)

# Predict
preds = model.predict(test_gen)
preds = (preds > 0.5).astype(int).reshape(-1)

# True labels
true_labels = test_gen.classes
class_names = list(test_gen.class_indices.keys())

# Report
report = classification_report(true_labels, preds, target_names=class_names)
print(" Multilingual test classification report:\n")
print(report)
