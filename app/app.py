import os
import json
import re

import torch
import torch.nn as nn

from flask import Flask, render_template, request, jsonify


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "templates"
)

STATIC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "static"
)

VOCAB_FILE = os.path.join(
    DATA_DIR,
    "vocabulary_final.json"
)

BASELINE_MODEL_FILE = os.path.join(
    DATA_DIR,
    "gender_bilstm_baseline.pth"
)

REGULARIZED_MODEL_FILE = os.path.join(
    DATA_DIR,
    "gender_bilstm_regularized.pth"
)


app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


MAX_LENGTH = 50

BASELINE_WEIGHT = 0.50
REGULARIZED_WEIGHT = 0.50


print("=" * 70)
print("INDIAN NAME GENDER CLASSIFIER")
print("=" * 70)

print("\nBase directory:")
print(BASE_DIR)

print("\nData directory:")
print(DATA_DIR)

print("\nLoading vocabulary...")


if not os.path.exists(VOCAB_FILE):
    raise FileNotFoundError(
        f"Vocabulary file not found:\n{VOCAB_FILE}"
    )

with open(
    VOCAB_FILE,
    "r",
    encoding="utf-8"
) as f:
    vocabulary = json.load(f)


# Support different vocabulary formats

if "char_to_id" in vocabulary:
    char_to_id = vocabulary["char_to_id"]

elif "char_to_index" in vocabulary:
    char_to_id = vocabulary["char_to_index"]

else:
    char_to_id = vocabulary


print(
    f"Vocabulary size: {len(char_to_id)}"
)


PAD_ID = char_to_id.get(
    "<PAD>",
    0
)

UNK_ID = char_to_id.get(
    "<UNK>",
    1
)


class GenderBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        hidden_dim=64,
        dropout=0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=PAD_ID
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc = nn.Linear(
            hidden_dim * 2,
            1
        )

    def forward(self, x):
        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(
            embedded
        )

        forward_hidden = hidden[-2]

        backward_hidden = hidden[-1]

        combined = torch.cat(
            (
                forward_hidden,
                backward_hidden
            ),
            dim=1
        )

        combined = self.dropout(
            combined
        )

        logits = self.fc(
            combined
        )

        return logits.squeeze(1)


print("\nLoading models...")

vocab_size = len(char_to_id)


baseline_model = GenderBiLSTM(
    vocab_size=vocab_size,
    embedding_dim=64,
    hidden_dim=64,
    dropout=0.3
)


regularized_model = GenderBiLSTM(
    vocab_size=vocab_size,
    embedding_dim=64,
    hidden_dim=64,
    dropout=0.3
)


if not os.path.exists(BASELINE_MODEL_FILE):
    raise FileNotFoundError(
        f"Baseline model not found:\n"
        f"{BASELINE_MODEL_FILE}"
    )


if not os.path.exists(REGULARIZED_MODEL_FILE):
    raise FileNotFoundError(
        f"Regularized model not found:\n"
        f"{REGULARIZED_MODEL_FILE}"
    )


baseline_state = torch.load(
    BASELINE_MODEL_FILE,
    map_location=DEVICE,
    weights_only=True
)


if (
    isinstance(baseline_state, dict)
    and "model_state_dict" in baseline_state
):
    baseline_state = baseline_state[
        "model_state_dict"
    ]


baseline_model.load_state_dict(
    baseline_state
)


regularized_state = torch.load(
    REGULARIZED_MODEL_FILE,
    map_location=DEVICE,
    weights_only=True
)


if (
    isinstance(regularized_state, dict)
    and "model_state_dict" in regularized_state
):
    regularized_state = regularized_state[
        "model_state_dict"
    ]


regularized_model.load_state_dict(
    regularized_state
)


baseline_model.to(DEVICE)
regularized_model.to(DEVICE)

baseline_model.eval()
regularized_model.eval()


print("[OK] Baseline Bi-LSTM loaded")
print("[OK] Regularized Bi-LSTM loaded")
print(f"[OK] Device: {DEVICE}")


def clean_name(name):
    name = str(name)

    name = name.strip()

    name = name.lower()

    name = re.sub(
        r"\s+",
        " ",
        name
    )

    return name


def encode_name(name):
    name = clean_name(name)

    encoded = []

    for char in name:
        encoded.append(
            char_to_id.get(
                char,
                UNK_ID
            )
        )

    encoded = encoded[:MAX_LENGTH]

    if len(encoded) < MAX_LENGTH:
        encoded += [
            PAD_ID
        ] * (
            MAX_LENGTH - len(encoded)
        )

    return torch.tensor(
        [encoded],
        dtype=torch.long,
        device=DEVICE
    )


@torch.no_grad()
def predict_name(name):
    cleaned_name = clean_name(name)

    if not cleaned_name:
        raise ValueError(
            "Please enter a name."
        )

    encoded = encode_name(
        cleaned_name
    )

    baseline_logits = baseline_model(
        encoded
    )

    baseline_probability = torch.sigmoid(
        baseline_logits
    ).item()

    regularized_logits = regularized_model(
        encoded
    )

    regularized_probability = torch.sigmoid(
        regularized_logits
    ).item()

    ensemble_probability = (
        BASELINE_WEIGHT
        * baseline_probability
        +
        REGULARIZED_WEIGHT
        * regularized_probability
    )

    male_probability = ensemble_probability

    female_probability = (
        1.0 - male_probability
    )

    if male_probability >= 0.5:
        predicted_gender = "Male"

        confidence = male_probability

    else:
        predicted_gender = "Female"

        confidence = female_probability

    confidence_percent = (
        confidence * 100
    )

    if confidence_percent >= 90:
        confidence_level = "High"

    elif confidence_percent >= 70:
        confidence_level = "Moderate"

    else:
        confidence_level = "Low"

    result = {
        "name": cleaned_name,
        "prediction": predicted_gender,
        "male_probability": round(
            male_probability * 100,
            2
        ),
        "female_probability": round(
            female_probability * 100,
            2
        ),
        "confidence": round(
            confidence_percent,
            2
        ),
        "confidence_level": confidence_level,
        "baseline_male_probability": round(
            baseline_probability * 100,
            2
        ),
        "regularized_male_probability": round(
            regularized_probability * 100,
            2
        )
    }

    print("\n" + "-" * 70)

    print("PREDICTION")

    print("-" * 70)

    print(
        f"Input name:              {name}"
    )

    print(
        f"Cleaned name:            {cleaned_name}"
    )

    print(
        f"Baseline male prob:      "
        f"{result['baseline_male_probability']:.2f}%"
    )

    print(
        f"Regularized male prob:   "
        f"{result['regularized_male_probability']:.2f}%"
    )

    print(
        f"Ensemble male prob:      "
        f"{result['male_probability']:.2f}%"
    )

    print(
        f"Ensemble female prob:    "
        f"{result['female_probability']:.2f}%"
    )

    print(
        f"Prediction:              "
        f"{result['prediction']}"
    )

    print(
        f"Confidence:              "
        f"{result['confidence']:.2f}%"
    )

    print(
        f"Confidence level:        "
        f"{result['confidence_level']}"
    )

    print("-" * 70)

    return result


@app.route(
    "/",
    methods=["GET"]
)
def home():
    return render_template(
        "index.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():
    try:
        print("\nReceived prediction request")

        if request.is_json:
            data = request.get_json(
                silent=True
            )

            print(
                f"Request JSON: {data}"
            )

            if not isinstance(
                data,
                dict
            ):
                raise ValueError(
                    "Invalid JSON request."
                )

            name = data.get(
                "name",
                ""
            )

        else:
            name = request.form.get(
                "name",
                ""
            )

            print(
                f"Request form name: {name}"
            )

        result = predict_name(
            name
        )

        # Return result directly (flattened) to match frontend expectations and prevent undefined errors

        if request.is_json:
            response = {
                "success": True,
                **result
            }

            print(
                f"API response: {response}"
            )

            return jsonify(
                response
            )

        return render_template(
            "index.html",
            result=result
        )

    except Exception as e:
        print(
            "\n!!! PREDICTION ERROR !!!"
        )

        print(
            f"Error type: {type(e).__name__}"
        )

        print(
            f"Error message: {e}"
        )

        if request.is_json:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 400

        return render_template(
            "index.html",
            error=str(e)
        )


@app.route(
    "/health",
    methods=["GET"]
)
def health():
    return jsonify({
        "status": "healthy",
        "model":
            "50/50 Bi-LSTM Ensemble",
        "device":
            str(DEVICE),
        "vocabulary_size":
            vocab_size,
        "maximum_sequence_length":
            MAX_LENGTH
    })


if __name__ == "__main__":
    print("\n" + "=" * 70)

    print(
        "SERVER STARTING"
    )

    print("=" * 70)

    print(
        "\nOpen:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "\nHealth check:"
    )

    print(
        "http://127.0.0.1:5000/health"
    )

    print(
        "\nModel:"
    )

    print(
        "50/50 Baseline Bi-LSTM + Regularized Bi-LSTM"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Vocabulary: {vocab_size}"
    )

    print(
        "\n"
    )

    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )