import torch
import torch.nn as nn


class GenderBiLSTM(nn.Module):
    """
    Bi-LSTM architecture used by the final project models.

    Architecture:
        Embedding
        ↓
        Bidirectional LSTM
        ↓
        Dropout
        ↓
        Fully Connected Layer
    """

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
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_dim * 2,
            1
        )

    def forward(self, x):
        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(embedded)

        # Forward final hidden state
        forward_hidden = hidden[-2]

        # Backward final hidden state
        backward_hidden = hidden[-1]

        # Combine both directions
        combined = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1
        )

        combined = self.dropout(combined)

        logits = self.fc(combined)

        return logits