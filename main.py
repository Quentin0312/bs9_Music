import streamlit as st

import librosa
import numpy as np
import pandas as pd
import joblib


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


def audio_to_csv(audio, scaler):
    def audio_pipeline(audio):
        features = []

        # Calcul du ZCR (Zero Crossing Rate)
        zcr = librosa.zero_crossings(audio)
        features.append(np.sum(zcr))  # Custom
        features.append(np.mean(zcr))
        features.append(np.var(zcr))

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

    # Load the audio file
    audio, _ = librosa.load(audio, sr=None)

    # Perform audio feature extraction
    features = audio_pipeline(audio)

    # Scale the features using the loaded scaler
    scaled_features = scaler.transform([features])

    # Create a DataFrame
    column_names = [
        "zcr_sum",
        "zcr_mean",
        "zcr_var",
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

    df = pd.DataFrame([features], columns=column_names)

    return df, scaled_features


# Streamlit app
st.title("Prédiction genre musical")

# Add interactive components for file upload
uploaded_file = st.file_uploader("Télécharger un fichier audio", type=["wav"])

if uploaded_file is not None:
    # Load the trained model
    # ! Load pytorch model
    # model = keras.models.load_model("./mymodelv1.keras")

    # Load the StandardScaler used during training
    scaler = joblib.load(
        "standard_scaler_pytorch_model.pkl"
    )  # Load the scaler from the saved file

    # Perform audio processing and get DataFrame and scaled features
    # ! Cut the audio with 3 sec samples
    df, scaled_features = audio_to_csv(uploaded_file, scaler)

    # Display the DataFrame
    st.write("Audio Features:")
    st.write(df)

    # Perform model prediction
    # prediction = model.predict(scaled_features)

    # predicted_genres = [genre_mapping[i] for i in range(len(prediction[0]))]

    # Display prediction result
    # st.write("Prediction:")
    # st.write(pd.DataFrame([prediction[0]], columns=predicted_genres))
