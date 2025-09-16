# train_cnn.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np

# Paths
train_dir = r"G:\final Audiodeepfake\processed_spectrograms\english"

# Data generators with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

img_size = (128, 128)
batch_size = 32

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Smaller CNN model with dropout
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# Evaluate on validation set
val_gen.reset()
preds = model.predict(val_gen)
preds = (preds > 0.5).astype(int).reshape(-1)

true_labels = val_gen.classes

# Classification report
report = classification_report(true_labels, preds, target_names=list(val_gen.class_indices.keys()))
print(" Final classification report:\n")
print(report)

# Save model
model.save(r"G:\final Audiodeepfake\cnn_model_smaller.keras")
print(" Model saved at G:\\final Audiodeepfake\\cnn_model_smaller.keras")
