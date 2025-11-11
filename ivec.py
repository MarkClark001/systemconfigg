# --- Install dependencies ---
# pip install librosa scikit-learn numpy python_speech_features

import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from python_speech_features import mfcc

# --- 1️⃣ Feature extraction (MFCCs) ---
def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=16000)
    mfcc_feat = mfcc(signal, sr, numcep=20, nfft=1200)
    return np.mean(mfcc_feat, axis=0)

# --- 2️⃣ Load dataset ---
# Folder structure:
# speakers/
# ├── speaker1/
# │   ├── s1_1.wav
# │   ├── s1_2.wav
# ├── speaker2/
# │   ├── s2_1.wav
# │   ├── s2_2.wav

dataset_path = "speakers"
X = []
y = []

for speaker in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, speaker)
    if os.path.isdir(folder):
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                feat = extract_features(os.path.join(folder, file))
                X.append(feat)
                y.append(speaker)

X = np.array(X)
y = np.array(y)
print("✅ Loaded features from", len(set(y)), "speakers.")

# --- 3️⃣ Universal Background Model (UBM) ---
# Trains a GMM to represent general voice characteristics of all speakers
ubm = GaussianMixture(n_components=8, covariance_type='diag', max_iter=200)
ubm.fit(X)

# --- 4️⃣ Compute i-vectors (simplified) ---
# Here we’ll just use GMM responsibilities as a toy “i-vector” representation
def compute_ivector(feat):
    posterior = ubm.predict_proba([feat])[0]
    return posterior

ivectors = np.array([compute_ivector(f) for f in X])

# --- 5️⃣ Optionally reduce dimension (LDA) ---
lda = LinearDiscriminantAnalysis(n_components=min(len(set(y)) - 1, ivectors.shape[1]-1))
ivectors_lda = lda.fit_transform(ivectors, y)

# --- 6️⃣ Train speaker classifier (SVM) ---
clf = SVC(kernel='linear', probability=True)
clf.fit(ivectors_lda, y)

# --- 7️⃣ Test on a new sample ---
test_file = "test_speaker.wav"
test_feat = extract_features(test_file)
test_ivector = compute_ivector(test_feat).reshape(1, -1)
test_ivector_lda = lda.transform(test_ivector)
pred = clf.predict(test_ivector_lda)[0]
print("🎤 Predicted Speaker:", pred)
