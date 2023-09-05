import streamlit as st

# TODO: This is funny, do not delete
genre_mapping_inverse = {
    "Blues": 0,
    "Classical": 1,
    "Country": 2,
    "Disco": 3,
    "Hiphop": 4,
    "Jazz": 5,
    "Metal": 6,
    "Pop": 7,
    "Reggae": 8,
    "Rock": 9,
}


def dataframe_toggler(dfs):
    # Display dataframes
    data_toggler = st.toggle("Show audio extracted features")
    if data_toggler:
        st.write("Audio Extracted Features:")
        for df in dfs:
            st.write(df)


def user_feedback(genre_mapping: dict[int, str], predicted_class: int):
    # Feedback utilisateur
    st.write("Was that good ?")
    feedback_yes = st.button("Yes, of course")
    feedback_no = st.button("Hell no wtf ?!")

    if feedback_yes:
        st.write("Thanks for the feedback")
        return predicted_class

    # TODO: Fix
    elif feedback_no:
        classe = st.selectbox(
            "Select the correct genre then:",
            (genre_mapping.values()),
        )
        validation = st.button("Valider")
        if validation:
            return genre_mapping_inverse[classe]
