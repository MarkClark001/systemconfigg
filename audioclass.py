# --- Install required packages (run in notebook or terminal) ---
# pip install librosa python_speech_features spafe numpy scipy

import numpy as np
import librosa
from python_speech_features import mfcc, logfbank
from spafe.features.lpc import lpcc
from spafe.features.gfcc import gfcc
from spafe.features.plp import plp

# --- 1️⃣ Load an audio file ---
# Replace with your own WAV file (mono, 16kHz recommended)
file_path = "sample_emotion.wav"
signal, sr = librosa.load(file_path, sr=16000)

# --- 2️⃣ Feature Extraction ---

# MFCC
mfcc_feat = mfcc(signal, samplerate=sr, numcep=13)
mfcc_mean = np.mean(mfcc_feat, axis=0)

# LPCC
lpcc_feat = lpcc.lpc2lpcc(signal, order=13)
lpcc_mean = np.mean(lpcc_feat, axis=0)

# PLP
plp_feat = plp.plp(signal, fs=sr, num_ceps=13)
plp_mean = np.mean(plp_feat, axis=0)

# GFCC
gfcc_feat = gfcc.gfcc(signal, fs=sr, num_ceps=13)
gfcc_mean = np.mean(gfcc_feat, axis=0)

# --- 3️⃣ Combine all features ---
features = np.hstack([mfcc_mean, lpcc_mean, plp_mean, gfcc_mean])
print("Feature vector shape:", features.shape)
print(features)


# Example: simple classifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Suppose X = all features, y = emotion labels
X = np.array([features])  # you’d normally have many samples
y = np.array([1])         # e.g. 1 = happy, 0 = sad, etc.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
print("Predicted:", clf.predict(X_test))


