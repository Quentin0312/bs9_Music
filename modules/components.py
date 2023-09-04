import streamlit as st


def dataframe_toggler(dfs):
    # Display dataframes
    data_toggler = st.toggle("Show audio extracted features")
    if data_toggler:
        st.write("Audio Extracted Features:")
        for df in dfs:
            st.write(df)
