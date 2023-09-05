import streamlit as st


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
            options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
        return classe
