"""
app.py — local Flask server
Run: python app.py  →  open http://localhost:5000

No NLTK, no PyTorch needed.
"""

import os, re, pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
STATIC     = os.path.join(BASE, "static")
VEC_PATH   = os.path.join(BASE, "vectorizer.pkl")
MODEL_PATH = os.path.join(BASE, "model_weights.npz")


# ── Stopwords (no NLTK needed) ────────────────────────────────────────────────
STOPWORDS = frozenset("""
i me my myself we our ours ourselves you your yours yourself yourselves
he him his himself she her hers herself it its itself they them their
theirs themselves what which who whom this that these those am is are
was were be been being have has had having do does did doing a an the
and but if or because as until while of at by for with about against
between into through during before after above below to from up down
in out on off over under again further then once here there when where
why how all both each few more most other some such no nor not only
own same so than too very s t can will just don should now d ll m o
re ve y ain aren couldn didn doesn don hadn hasn haven isn mightn
mustn needn shan shouldn wasn weren won wouldn
""".split())

# ── Regex stemmer (no NLTK needed) ────────────────────────────────────────────
_RULES = [
    (r'ational$','ate'),(r'tional$','tion'),(r'enci$','ence'),(r'anci$','ance'),
    (r'izer$','ize'),(r'izing$','ize'),(r'ising$','ise'),(r'ation$','ate'),
    (r'ator$','ate'),(r'alism$','al'),(r'iveness$','ive'),(r'fulness$','ful'),
    (r'ousness$','ous'),(r'aliti$','al'),(r'iviti$','ive'),(r'biliti$','ble'),
    (r'icate$','ic'),(r'alize$','al'),(r'iciti$','ic'),(r'ical$','ic'),
    (r'ful$',''),(r'ness$',''),(r'ement$',''),(r'ment$',''),(r'ent$',''),
    (r'ism$',''),(r'ate$',''),(r'iti$',''),(r'ous$',''),(r'ive$',''),
    (r'ize$',''),(r'al$',''),(r'ing$',''),(r'ings$',''),(r'ied$','i'),
    (r'ies$','i'),(r'ed$',''),(r'ly$',''),(r'er$',''),(r'est$',''),
    (r'ion$',''),(r'tion$',''),(r'able$',''),(r'ible$',''),(r'ant$',''),
    (r'ance$',''),(r'ence$',''),(r'ary$',''),(r'ory$',''),
]
_COMPILED = [(re.compile(p + '$'), r) for p, r in _RULES]

def _stem(word):
    if len(word) <= 3:
        return word
    for pat, rep in _COMPILED:
        c = pat.sub(rep, word)
        if c != word and len(c) >= 3:
            return c
    return word

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+',      '', text)
    text = re.sub(r'<.*?>',        '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return ' '.join(_stem(t) for t in text.split()
                    if t not in STOPWORDS and len(t) > 1)


# ── Numpy RNN ─────────────────────────────────────────────────────────────────
class NumpyRNN:
    def __init__(self, path):
        w         = np.load(path)
        self.W_ih = w["W_ih"]
        self.W_hh = w["W_hh"]
        self.b_ih = w["b_ih"]
        self.b_hh = w["b_hh"]
        self.W_fc = w["W_fc"]
        self.b_fc = w["b_fc"]
        self.H    = self.W_hh.shape[0]

    def predict(self, x):
        h = np.zeros(self.H, dtype=np.float32)
        h = np.tanh(self.W_ih @ x + self.b_ih + self.W_hh @ h + self.b_hh)
        logit = (self.W_fc @ h + self.b_fc)[0]
        return float(1.0 / (1.0 + np.exp(-logit)))


# ── Load model & vectorizer ───────────────────────────────────────────────────
model      = None
vectorizer = None

try:
    if os.path.exists(VEC_PATH):
        with open(VEC_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        print("✓  vectorizer loaded")
    else:
        print("⚠  vectorizer.pkl not found")

    if os.path.exists(MODEL_PATH) and vectorizer is not None:
        model = NumpyRNN(MODEL_PATH)
        print("✓  model loaded")
    else:
        print("⚠  model_weights.npz not found — using rule-based fallback")
        print("   Run: python export_model.py")
except Exception as e:
    print(f"⚠  Error loading model: {e}")


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")

@app.route("/health")
def health():
    return jsonify({
        "status":            "ok",
        "model_loaded":      model is not None,
        "vectorizer_loaded": vectorizer is not None,
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    processed = preprocess(text)

    # Rule-based fallback
    if model is None or vectorizer is None:
        POS = {"good","great","excel","amaz","wonder","fantast","love","best",
               "brilliant","outstand","superb","beauti","enjoy","perfect",
               "incred","awesom","terrif","masterpiec","hilari","touch",
               "inspir","entertain","delight"}
        NEG = {"bad","terribl","awful","worst","horribl","hate","bore","dull",
               "wast","disappoint","poor","mediocr","dread","annoy","stupid",
               "pathet","useless","garbage","trash","disast","rubbish","ridicul"}
        words = set(processed.split())
        p, n  = len(words & POS), len(words & NEG)
        total = p + n or 1
        label = "positive" if p >= n else "negative"
        conf  = round(min(50 + abs(p - n) / total * 50, 99), 1)
        return jsonify({"sentiment": label, "confidence": conf, "model_loaded": False})

    # RNN inference
    x     = vectorizer.transform([processed]).toarray()[0].astype(np.float32)
    prob  = model.predict(x)
    label = "positive" if prob >= 0.5 else "negative"
    conf  = round((prob if prob >= 0.5 else 1 - prob) * 100, 1)
    return jsonify({"sentiment": label, "confidence": conf, "model_loaded": True})


if __name__ == "__main__":
    # Sanity check
    if not os.path.exists(STATIC):
        print(f"\n✗  ERROR: 'static/' folder not found at {STATIC}")
        print("   Create a 'static/' folder and put index.html inside it.\n")
    elif not os.path.exists(os.path.join(STATIC, "index.html")):
        print(f"\n✗  ERROR: static/index.html not found")
        print("   Make sure index.html is inside the 'static/' folder.\n")
    else:
        print("\n✓  static/index.html found")

    print("✓  Starting server →  http://localhost:5000\n")
    app.run(debug=True, port=5000)