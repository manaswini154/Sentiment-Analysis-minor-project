"""
app.py — Flask backend + serves the UI
Runs inference with plain numpy (no PyTorch needed).

Run locally:  python app.py
Deploy:       push to Render / Railway (python app.py as start command)
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import re, pickle, os, numpy as np

# ── NLTK (tiny, ~3 MB) ───────────────────────────────────────────────────────
import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords
from nltk.stem     import PorterStemmer


# ── Pure-numpy RNN forward pass ───────────────────────────────────────────────
class NumpyRNN:
    """
    Mirrors the PyTorch RNN exactly:
      - single-layer Elman RNN, tanh activation
      - followed by a linear (fc) layer
      - sigmoid output
    """
    def __init__(self, weights_path: str):
        w = np.load(weights_path)
        self.W_ih = w["W_ih"]   # (hidden, input)
        self.W_hh = w["W_hh"]   # (hidden, hidden)
        self.b_ih = w["b_ih"]   # (hidden,)
        self.b_hh = w["b_hh"]   # (hidden,)
        self.W_fc = w["W_fc"]   # (1, hidden)
        self.b_fc = w["b_fc"]   # (1,)
        self.hidden_size = self.W_hh.shape[0]

    def forward(self, x: np.ndarray) -> float:
        """
        x: shape (input_size,)  — one TF-IDF vector
        returns: probability in [0, 1]
        """
        h = np.zeros(self.hidden_size, dtype=np.float32)
        # single time-step (seq_len=1, same as training unsqueeze(1))
        h = np.tanh(self.W_ih @ x + self.b_ih + self.W_hh @ h + self.b_hh)
        logit = self.W_fc @ h + self.b_fc          # shape (1,)
        prob  = 1.0 / (1.0 + np.exp(-logit[0]))   # sigmoid
        return float(prob)


# ── Text preprocessing ────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+",         "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"<.*?>",           "", text)
    tokens     = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    ps         = PorterStemmer()
    return " ".join(ps.stem(t) for t in tokens if t not in stop_words)


# ── Load artifacts at startup ─────────────────────────────────────────────────
model      = None
vectorizer = None

if os.path.exists("vectorizer.pkl"):
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("✓  vectorizer loaded")

if os.path.exists("model_weights.npz") and vectorizer is not None:
    model = NumpyRNN("model_weights.npz")
    print("✓  model weights loaded (numpy)")
else:
    print("⚠  model_weights.npz not found — using rule-based fallback")
    print("   Run: python export_model.py")


# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/health")
def health():
    return jsonify({
        "status":           "ok",
        "model_loaded":     model is not None,
        "vectorizer_loaded": vectorizer is not None,
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    processed = preprocess(text)

    # ── Rule-based fallback ───────────────────────────────────────────────
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
        return jsonify({"sentiment": label, "confidence": confidence, "model_loaded": False})

    # ── Numpy RNN inference ───────────────────────────────────────────────
    x    = vectorizer.transform([processed]).toarray()[0].astype(np.float32)
    prob = model.forward(x)

    label      = "positive" if prob >= 0.5 else "negative"
    confidence = round((prob if prob >= 0.5 else 1 - prob) * 100, 1)

    return jsonify({"sentiment": label, "confidence": confidence, "model_loaded": True})


if __name__ == "__main__":
    print("\n✓  Open http://localhost:5000\n")
    app.run(debug=True, port=5000)