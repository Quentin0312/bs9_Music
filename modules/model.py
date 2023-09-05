from torch import nn
import torch

import streamlit as st

"""
No device agnostic code because CUDA
will not be available in deployement environnement
"""

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


class MusicClassifier(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(
                in_features=input_features, out_features=256, dtype=torch.float32
            ),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256, out_features=128, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=128, out_features=output_features, dtype=torch.float32
            ),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


def predict(dfs):
    # Load the trained model
    my_model = MusicClassifier(input_features=55, output_features=10)
    my_model.load_state_dict(
        torch.load(
            f="./resources/actual_model_fast.pth", map_location=torch.device("cpu")
        )
    )

    # TODO Rewrite
    # Evaluation mode
    my_model.eval()

    class_predictions = []
    for df in dfs:
        y_logits = my_model(torch.from_numpy(df.to_numpy()).type(torch.float32))
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        # st.write(genre_mapping[y_pred.detach().numpy()[0]])
        class_predictions.append(genre_mapping[y_pred.detach().numpy()[0]])

    unique_values = set(class_predictions)
    actual_best = 0
    for elt in unique_values:
        if class_predictions.count(elt) > actual_best:
            prediction = elt
        st.write(elt, class_predictions.count(elt))

    return genre_mapping_inverse[prediction]
