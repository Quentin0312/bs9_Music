import streamlit as st

import torch
from torch import nn

import librosa
import joblib

import numpy as np
import pandas as pd

from typing import List

# ! Device agnostic code ?


class MusicClassifier(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(
                in_features=input_features, out_features=2048, dtype=torch.float32
            ),
            nn.GELU(),
            nn.Dropout(p=0.6),
            nn.Linear(in_features=2048, out_features=1024, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(p=0.6),
            nn.Linear(in_features=1024, out_features=512, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(p=0.6),
            nn.Linear(in_features=512, out_features=256, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(p=0.6),
            nn.Linear(in_features=256, out_features=128, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(p=0.6),
            nn.Linear(in_features=128, out_features=64, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(p=0.6),
            nn.Linear(
                in_features=64, out_features=output_features, dtype=torch.float32
            ),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


# Create a mapping from numerical values to genre names
genre_mapping = {
    0: "Blues",
    1: "Classical",
    2: "Country",
    3: "Disco",
    4: "Hiphop",
    5: "Jazz",
    6: "Metal",
    7: "Pop",
    8: "Reggae",
    9: "Rock",
}


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
    # ! Vérif
    audio, sample_rate = librosa.load(audio, sr=None)

    segment_duration = 3  # Durée de chaque segment en secondes
    segment_length = int(sample_rate * segment_duration)
    segments = []

    # Effectuez la prédiction toutes les 3 secondes
    for i in range(0, len(audio), segment_length):
        segment = audio[i : i + segment_length]
        segments.append(segment)

    return segments


# ! Refactoriser / découper !
def audio_to_csv(audio, scaler) -> List[pd.DataFrame]:
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


# Streamlit app
st.title("Prédiction genre musical")

# Add interactive components for file upload
uploaded_file = st.file_uploader("Télécharger un fichier audio", type=["wav"])

if uploaded_file is not None:
    # Load the StandardScaler used during training
    scaler = joblib.load(
        "standard_scaler_pytorch_model_last.pkl"
    )  # Load the scaler from the saved file

    # Perform audio processing and get DataFrame and scaled features
    dfs = audio_to_csv(uploaded_file, scaler)

    # Display the DataFrame
    data_toggler = st.toggle("Show audio extracted features")
    if data_toggler:
        st.write("Audio Extracted Features:")
        for df in dfs:
            st.write(df)

    # Load the trained model
    my_model = MusicClassifier(input_features=55, output_features=10)
    my_model.load_state_dict(
        torch.load(f="./my_pytorch_model.pth", map_location=torch.device("cpu"))
    )

    my_model.eval()
    for df in dfs:
        y_logits = my_model(torch.from_numpy(df.to_numpy()).type(torch.float32))
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        st.write(genre_mapping[y_pred.detach().numpy()[0]])
