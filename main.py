import streamlit as st

from modules.model import predict, genre_mapping
from modules.preprocessing import audio_to_csv
from modules.components import dataframe_toggler

# TODO Rename things
# TODO Check librosa printed warnings


# Streamlit app ------------------------------------------------------------------------
st.title("Prédiction genre musical")

# File upload
uploaded_file = st.file_uploader(
    "Télécharger un fichier audio", type=["wav"]
)  # TODO Add mp3 !?


if uploaded_file is not None:
    # Perform audio processing
    dfs = audio_to_csv(uploaded_file)

    # Afficher les inputs preprocessed
    dataframe_toggler(dfs)

    # Predictions
    predict(dfs)

    # TODO: Externalise component
    # Feedback utilisateur
    st.write("Was that good ?")
    if st.button("Yes, of course"):
        st.write("Thanks for the feedback")
    elif st.button("Hell no wtf ?!"):
        classe = st.selectbox(
            "Select the correct genre then:", (genre_mapping.values())
        )
        st.write(classe)

    # Nouvel entrainement (sytématique)
