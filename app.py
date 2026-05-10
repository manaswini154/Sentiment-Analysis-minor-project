"""
app.py — Flask backend + serves the UI
Run: python app.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import re, pickle, os, nltk, torch, torch.nn as nn, numpy as np

nltk.download("punkt",      quiet=True)
nltk.download("punkt_tab",  quiet=True)
nltk.download("stopwords",  quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords
from nltk.stem     import PorterStemmer

# ── Model (must match train.py) ───────────────────────────────────────────────
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0  = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+",    "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"<.*?>",      "", text)
    tokens     = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    ps         = PorterStemmer()
    tokens     = [ps.stem(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


# ── Load artifacts ────────────────────────────────────────────────────────────
model      = None
vectorizer = None

if os.path.exists("vectorizer.pkl"):
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

if os.path.exists("model.pt") and vectorizer:
    m = RNN(len(vectorizer.vocabulary_))
    m.load_state_dict(torch.load("model.pt", map_location="cpu"))
    m.eval()
    model = m


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    processed = preprocess(text)

    # ── Rule-based fallback if model not loaded ───────────────────────────
    if model is None or vectorizer is None:
        pos_words = {"good","great","excellent","amazing","wonderful","fantastic",
                     "love","best","brilliant","outstanding","superb","beautiful",
                     "enjoy","perfect","incredible","awesome","terrific","masterpiece",
                     "hilarious","touching","inspiring","entertaining","delightful"}
        neg_words = {"bad","terrible","awful","worst","horrible","hate","boring",
                     "dull","waste","disappointing","poor","mediocre","dreadful",
                     "annoying","stupid","pathetic","useless","garbage","trash",
                     "disaster","rubbish","ridiculous"}
        words = set(processed.split())
        pos   = len(words & pos_words)
        neg   = len(words & neg_words)
        total = pos + neg or 1
        if pos >= neg:
            label      = "positive"
            confidence = round(min(50 + (pos - neg) / total * 50, 99), 1)
        else:
            label      = "negative"
            confidence = round(min(50 + (neg - pos) / total * 50, 99), 1)
        return jsonify({
            "sentiment":    label,
            "confidence":   confidence,
            "model_loaded": False
        })

    # ── Real RNN inference ────────────────────────────────────────────────
    X      = vectorizer.transform([processed]).toarray()
    tensor = torch.from_numpy(X).float().unsqueeze(1)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor).squeeze()).item()

    label      = "positive" if prob >= 0.5 else "negative"
    confidence = round((prob if prob >= 0.5 else 1 - prob) * 100, 1)

    return jsonify({
        "sentiment":    label,
        "confidence":   confidence,
        "model_loaded": True
    })


if __name__ == "__main__":
    print("\n✓  Open http://localhost:5000 in your browser\n")
    if model is None:
        print("⚠  model.pt / vectorizer.pkl not found — using rule-based fallback")
        print("   Run  python train.py  first to train the model.\n")
    app.run(debug=True, port=5000)
