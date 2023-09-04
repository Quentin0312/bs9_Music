import librosa

import joblib

import numpy as np
import pandas as pd

from typing import List

# Load the StandardScaler used during training
scaler = joblib.load("standard_scaler_pytorch_model_last.pkl")


def audio_to_csv(audio) -> List[pd.DataFrame]:
    dfs = []
    segments = get_3sec_sample(audio)

    for audio in segments:
        # Perform audio feature extraction
        features = audio_pipeline(audio)

        # Scale the features using the loaded scaler
        scaled_features = scaler.transform([features])

        # Create a DataFrame
        column_names = [
            "chroma_mean",
            "chroma_var",
            "rms_mean",
            "rms_var",
            "spectral_c_mean",
            "spectral_c_var",
            "spectral_bandwith_mean",
            "spectral_bandwith_var",
            "rolloff_mean",
            "rolloff_var",
            # "zcr_sum",
            "zcr_mean",
            "zcr_var",
            "harmonic_mean",
            "harmonic_var",
            "tempo",
            "mfcc1_mean",
            "mfcc1_var",
            "mfcc2_mean",
            "mfcc2_var",
            "mfcc3_mean",
            "mfcc3_var",
            "mfcc4_mean",
            "mfcc4_var",
            "mfcc5_mean",
            "mfcc5_var",
            "mfcc6_mean",
            "mfcc6_var",
            "mfcc7_mean",
            "mfcc7_var",
            "mfcc8_mean",
            "mfcc8_var",
            "mfcc9_mean",
            "mfcc9_var",
            "mfcc10_mean",
            "mfcc10_var",
            "mfcc11_mean",
            "mfcc11_var",
            "mfcc12_mean",
            "mfcc12_var",
            "mfcc13_mean",
            "mfcc13_var",
            "mfcc14_mean",
            "mfcc14_var",
            "mfcc15_mean",
            "mfcc15_var",
            "mfcc16_mean",
            "mfcc16_var",
            "mfcc17_mean",
            "mfcc17_var",
            "mfcc18_mean",
            "mfcc18_var",
            "mfcc19_mean",
            "mfcc19_var",
            "mfcc20_mean",
            "mfcc20_var",
        ]

        df = pd.DataFrame(scaled_features, columns=column_names)
        dfs.append(df)

    return dfs


def audio_pipeline(audio):
    features = []

    # Chromagram
    chroma_stft = librosa.feature.chroma_stft(y=audio)
    features.append(np.mean(chroma_stft))
    features.append(np.var(chroma_stft))  # var => variance

    # RMS (Root Mean Square value for each frame)
    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))
    features.append(np.var(rms))

    # Calcul du Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio)
    features.append(np.mean(spectral_centroids))
    features.append(np.var(spectral_centroids))

    # Spectral bandwith
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio)
    features.append(np.mean(spectral_bandwidth))
    features.append(np.var(spectral_bandwidth))

    # Calcul du spectral rolloff point
    rolloff = librosa.feature.spectral_rolloff(y=audio)
    features.append(np.mean(rolloff))
    features.append(np.var(rolloff))

    # Calcul du ZCR (Zero Crossing Rate)
    zcr = librosa.zero_crossings(audio)
    # features.append(np.sum(zcr))  # Custom
    features.append(np.mean(zcr))
    features.append(np.var(zcr))

    # Harmonic
    harmony = librosa.effects.harmonic(y=audio)
    features.append(np.mean(harmony))
    features.append(np.var(harmony))

    # Tempo
    tempo = librosa.feature.tempo(y=audio)
    features.append(tempo[0])

    # Calcul des moyennes des MFCC
    mfcc = librosa.feature.mfcc(y=audio)
    for x in mfcc:
        features.append(np.mean(x))
        features.append(np.var(x))

    return features


def get_3sec_sample(audio) -> list:
    # TODO Vérif
    # TODO Rewrite
    audio, sample_rate = librosa.load(audio, sr=None)

    segment_duration = 3  # Durée de chaque segment en secondes
    segment_length = int(sample_rate * segment_duration)
    segments = []

    # Effectuez la prédiction toutes les 3 secondes
    for i in range(0, len(audio), segment_length):
        segment = audio[i : i + segment_length]
        segments.append(segment)

    return segments
