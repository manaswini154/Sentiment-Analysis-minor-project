"""
export_model.py — run this ONCE locally before deploying.

It converts model.pt (PyTorch) → model_weights.npz (plain numpy).
The server then only needs numpy, not torch.

Usage:
    python export_model.py
Produces:
    model_weights.npz   (~1 MB, commit this to your repo)
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
import os

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])


def export():
    if not os.path.exists("model.pt"):
        print("✗  model.pt not found. Run train.py first.")
        return
    if not os.path.exists("vectorizer.pkl"):
        print("✗  vectorizer.pkl not found. Run train.py first.")
        return

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    input_size = len(vectorizer.vocabulary_)
    model = RNN(input_size)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()

    sd = model.state_dict()

    # RNN weights: input→hidden and hidden→hidden (+ biases)
    # PyTorch RNN layer 0 key names:
    #   rnn.weight_ih_l0  shape (hidden, input)
    #   rnn.weight_hh_l0  shape (hidden, hidden)
    #   rnn.bias_ih_l0    shape (hidden,)
    #   rnn.bias_hh_l0    shape (hidden,)
    #   fc.weight         shape (1, hidden)
    #   fc.bias           shape (1,)

    np.savez(
        "model_weights.npz",
        W_ih = sd["rnn.weight_ih_l0"].numpy(),   # (128, input_size)
        W_hh = sd["rnn.weight_hh_l0"].numpy(),   # (128, 128)
        b_ih = sd["rnn.bias_ih_l0"].numpy(),      # (128,)
        b_hh = sd["rnn.bias_hh_l0"].numpy(),      # (128,)
        W_fc = sd["fc.weight"].numpy(),            # (1, 128)
        b_fc = sd["fc.bias"].numpy(),              # (1,)
    )

    size_mb = os.path.getsize("model_weights.npz") / 1e6
    print(f"✓  model_weights.npz saved  ({size_mb:.1f} MB)")
    print("   Commit this file to your repo. PyTorch not needed on the server.")


if __name__ == "__main__":
    export()