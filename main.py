import streamlit as st

from modules.model import predict, genre_mapping, MusicClassifier
from modules.preprocessing import audio_to_csv
from modules.components import dataframe_toggler, genre_mapping_inverse, user_feedback
from modules.training import concat_dfs, training_loop

# TODO Rename things
# TODO Check librosa printed warnings


# Streamlit app ------------------------------------------------------------------------
st.title("Prédiction genre musical")
# File upload
uploaded_file = st.file_uploader("Télécharger un fichier audio", type=["wav", "mp3"])


if uploaded_file is not None:
    # Perform audio processing
    dfs = audio_to_csv(uploaded_file)

    # Afficher les inputs preprocessed
    dataframe_toggler(dfs)

    # Predictions
    predicted_class = predict(dfs)

    # Feedback utilisateur
    submit, real_class = user_feedback(genre_mapping)

    if submit:
        # Création du nouveau dataset
        concat_dfs(dfs, genre_mapping_inverse[real_class])

        # Nouvel entrainement (sytématique)
        training_loop(MusicClassifier)
