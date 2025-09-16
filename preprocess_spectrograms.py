# preprocess_spectrograms.py

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ----------- CONFIGURE YOUR PATHS -----------
EN_REAL_DIR = r"G:\final Audiodeepfake\archive (1)\english\real"
EN_FAKE_DIR = r"G:\final Audiodeepfake\archive (1)\english\fake"
LANG_DIR = r"G:\final Audiodeepfake\lang"
OUTPUT_DIR = r"G:\final Audiodeepfake\processed_spectrograms"

# ----------- CREATE OUTPUT DIRS -----------
os.makedirs(os.path.join(OUTPUT_DIR, 'english', 'real'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'english', 'fake'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'lang'), exist_ok=True)

def process_audio(file_path, save_path, sr=16000):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(2, 2))
        librosa.display.specshow(S_DB, sr=sr)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f" Error processing {file_path}: {e}")

# ----------- PROCESS ENGLISH -----------
en_real_count = 0
en_fake_count = 0
lang_count = 0

for filename in os.listdir(EN_REAL_DIR):
    if filename.endswith('.wav'):
        input_path = os.path.join(EN_REAL_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, 'english', 'real', filename.replace('.wav', '.png'))
        process_audio(input_path, output_path)
        en_real_count += 1

for filename in os.listdir(EN_FAKE_DIR):
    if filename.endswith('.wav'):
        input_path = os.path.join(EN_FAKE_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, 'english', 'fake', filename.replace('.wav', '.png'))
        process_audio(input_path, output_path)
        en_fake_count += 1

# ----------- PROCESS LANGUAGES -----------
for lang in os.listdir(LANG_DIR):
    lang_path = os.path.join(LANG_DIR, lang)
    if os.path.isdir(lang_path):
        for label in ['real', 'fake']:
            label_path = os.path.join(lang_path, label)
            if os.path.isdir(label_path):
                out_dir = os.path.join(OUTPUT_DIR, 'lang', lang, label)
                os.makedirs(out_dir, exist_ok=True)

                for filename in os.listdir(label_path):
                    if filename.endswith('.wav'):
                        input_path = os.path.join(label_path, filename)
                        output_path = os.path.join(out_dir, filename.replace('.wav', '.png'))
                        process_audio(input_path, output_path)
                        lang_count += 1

# ----------- SUMMARY -----------
print(f" Processed {en_real_count} English real audio files.")
print(f" Processed {en_fake_count} English fake audio files.")
print(f" Processed {lang_count} multilingual (real + fake) audio files.")
print(f" All spectrograms saved in: {OUTPUT_DIR}")
