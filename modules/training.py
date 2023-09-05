# TODO: Use only absolute paths
# TODO: Move column_names

from typing import List
import pandas as pd
import streamlit as st

# TODO: Move and refactor
column_names = [
    "chroma_stft_mean",
    "chroma_stft_var",
    "rms_mean",
    "rms_var",
    "spectral_centroid_mean",
    "spectral_centroid_var",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_var",
    "rolloff_mean",
    "rolloff_var",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_var",
    "harmony_mean",
    "harmony_var",
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


def concat_dfs(dfs: List[pd.DataFrame], predicted_class):
    # Creating new df from processed input
    for i in range(len(dfs)):
        if i == 0:
            new_df = pd.DataFrame(dfs[i], columns=column_names)
        else:
            new_df = pd.concat([new_df, dfs[i]], axis=0)
    new_df["label"] = predicted_class
    st.write(new_df)

    # Créer dataset enrichie (csv)
    original_df = pd.read_csv(
        "/app/resources/original_dataset.csv"
    )  # ! Fait ça teh l'afaire
    # TODO Change to actual

    st.write(len(original_df))
    concatened_dataset = pd.concat([original_df, new_df], axis=0)
    st.write(len(concatened_dataset))
    st.write(concatened_dataset[9985:])

    concatened_dataset.to_csv("/app/resources/actual_dataset.csv")
