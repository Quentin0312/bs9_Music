import streamlit as st
import tensorflow as tf
import keras
import librosa
import numpy as np
import pandas as pd
import joblib

# import soundfile
# import audioread
from sklearn.preprocessing import StandardScaler as sc


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


def audio_to_csv(audio):
    def audio_pipeline(audio):
        features = []

        # Calcul du ZCR

        zcr = librosa.zero_crossings(audio)
        features.append(sum(zcr))

        # Calcul de la moyenne du Spectral centroid

        spectral_centroids = librosa.feature.spectral_centroid(y=audio)[0]
        features.append(np.mean(spectral_centroids))

        # Calcul du spectral rolloff point

        rolloff = librosa.feature.spectral_rolloff(y=audio)
        features.append(np.mean(rolloff))

        # Calcul des moyennes des MFCC

        mfcc = librosa.feature.mfcc(y=audio)

        for x in mfcc:
            features.append(np.mean(x))

        return features

    # Load the audio file
    audio, _ = librosa.load(audio, sr=None)

    # Perform audio feature extraction
    features = audio_pipeline(audio)

    # Scale the features using the loaded scaler
    scaled_features = scaler.transform([features])

    # Create a DataFrame
    column_names = [
        "zcr",
        "spectral_c",
        "rolloff",
        "mfcc1",
        "mfcc2",
        "mfcc3",
        "mfcc4",
        "mfcc5",
        "mfcc6",
        "mfcc7",
        "mfcc8",
        "mfcc9",
        "mfcc10",
        "mfcc11",
        "mfcc12",
        "mfcc13",
        "mfcc14",
        "mfcc15",
        "mfcc16",
        "mfcc17",
        "mfcc18",
        "mfcc19",
        "mfcc20",
    ]

    df = pd.DataFrame([features], columns=column_names)

    return df, scaled_features


# Streamlit app
st.title("Prédiction genre musical")

# Add interactive components for file upload
uploaded_file = st.file_uploader("Télécharger un fichier audio", type=["wav"])

if uploaded_file is not None:
    # Load the trained model
    model = keras.models.load_model("./mymodelv1.keras")

    # Load the StandardScaler used during training
    scaler = joblib.load("standard_scaler.pkl")  # Load the scaler from the saved file

    # Perform audio processing and get DataFrame and scaled features
    df, scaled_features = audio_to_csv(uploaded_file)

    # Display the DataFrame
    st.write("Audio Features:")
    st.write(df)

    # Perform model prediction
    prediction = model.predict(scaled_features)

    predicted_genres = [genre_mapping[i] for i in range(len(prediction[0]))]

    # Display prediction result
    st.write("Prediction:")
    st.write(pd.DataFrame([prediction[0]], columns=predicted_genres))
