import json
import os
import sys

import torch
import torch.nn.functional as F

# Allow importing model.py when running from project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model import GenderBiLSTM


BASE_DIR = os.path.dirname(CURRENT_DIR)

VOCAB_PATH = os.path.join(
    BASE_DIR,
    "data",
    "vocabulary_final.json"
)

BASELINE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "gender_bilstm_baseline.pth"
)

REGULARIZED_MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "gender_bilstm_regularized.pth"
)


DEVICE = torch.device("cpu")


with open(
    VOCAB_PATH,
    "r",
    encoding="utf-8"
) as f:
    vocabulary_data = json.load(f)


# Your vocabulary_final.json contains:
# {
#     "char_to_id": {...},
#     "max_length": 50
# }

CHAR_TO_ID = vocabulary_data["char_to_id"]

MAX_LENGTH = vocabulary_data.get(
    "max_length",
    50
)

VOCAB_SIZE = len(CHAR_TO_ID)


def load_model(model_path):
    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    # Your saved models may contain metadata.
    # Use the saved architecture parameters when available.

    embedding_dim = checkpoint.get(
        "embedding_dim",
        64
    )

    hidden_dim = checkpoint.get(
        "hidden_dim",
        64
    )

    dropout = checkpoint.get(
        "dropout",
        0.3
    )

    model = GenderBiLSTM(
        vocab_size=VOCAB_SIZE,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    # Handle both checkpoint formats:
    #
    # 1. {"model_state_dict": ...}
    # 2. direct state_dict

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)

    model.eval()

    return model


baseline_model = load_model(
    BASELINE_MODEL_PATH
)

regularized_model = load_model(
    REGULARIZED_MODEL_PATH
)


def normalize_name(name):
    if name is None:
        return ""

    name = str(name)

    # Match the character-level approach
    # used by the training pipeline.

    name = name.strip().lower()

    return name


def encode_name(name):
    name = normalize_name(name)

    encoded = []

    unk_id = CHAR_TO_ID.get(
        "<UNK>",
        1
    )

    pad_id = CHAR_TO_ID.get(
        "<PAD>",
        0
    )

    for char in name:
        char_id = CHAR_TO_ID.get(
            char,
            unk_id
        )

        encoded.append(char_id)

    encoded = encoded[:MAX_LENGTH]

    while len(encoded) < MAX_LENGTH:
        encoded.append(pad_id)

    return torch.tensor(
        [encoded],
        dtype=torch.long,
        device=DEVICE
    )


def get_model_probability(
    model,
    encoded
):
    with torch.no_grad():
        logits = model(
            encoded
        )

        probability = torch.sigmoid(
            logits
        ).item()

    return probability


def predict_name(name):
    name = normalize_name(name)

    if not name:
        return {
            "name": name,
            "prediction": "Unknown",
            "male_probability": 0.0,
            "female_probability": 0.0,
            "confidence": 0.0,
            "confidence_level": "Unknown"
        }

    encoded = encode_name(name)

    baseline_probability = get_model_probability(
        baseline_model,
        encoded
    )

    regularized_probability = get_model_probability(
        regularized_model,
        encoded
    )

    male_probability = (
        baseline_probability * 0.5
        +
        regularized_probability * 0.5
    )

    female_probability = (
        1.0 - male_probability
    )

    if male_probability >= 0.5:
        prediction = "Male"

        confidence = male_probability

    else:
        prediction = "Female"

        confidence = female_probability

    if confidence >= 0.90:
        confidence_level = "High"

    elif confidence >= 0.70:
        confidence_level = "Moderate"

    else:
        confidence_level = "Low"

    return {
        "name": name,
        "prediction": prediction,
        "male_probability": male_probability,
        "female_probability": female_probability,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "baseline_male_probability": baseline_probability,
        "regularized_male_probability": regularized_probability
    }