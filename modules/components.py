import streamlit as st


def dataframe_toggler(dfs):
    # Display dataframes
    data_toggler = st.toggle("Show audio extracted features")
    if data_toggler:
        st.write("Audio Extracted Features:")
        for df in dfs:
            st.write(df)


def user_feedback(genre_mapping: dict[int, str]):
    # Feedback utilisateur
    st.write("Was that good ?")
    feedback_yes = st.button("Yes, of course")
    feedback_no = st.button("Hell no wtf ?!")

    if feedback_yes:
        st.write("Thanks for the feedback")
        return None

    elif feedback_no:
        classe = st.selectbox(
            "Select the correct genre then:", (genre_mapping.values())
        )
        return classe
