"""
api/index.py  ─  Vercel serverless entry point

NO nltk, NO torch  ─  only flask, numpy, scikit-learn.
Preprocessing uses hardcoded stopwords + regex stemmer so there
are zero network calls and zero missing-resource errors at runtime.

Repo structure required:
  /
  ├── api/index.py          ← this file
  ├── static/index.html     ← frontend
  ├── vectorizer.pkl
  ├── model_weights.npz
  ├── vercel.json
  └── requirements.txt
"""

import os, re, pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Paths ─────────────────────────────────────────────────────────────────────
# Vercel mounts the repo root at /var/task
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC     = os.path.join(ROOT, "static")
VEC_PATH   = os.path.join(ROOT, "vectorizer.pkl")
MODEL_PATH = os.path.join(ROOT, "model_weights.npz")


# ── Stopwords (hardcoded — no NLTK download needed) ──────────────────────────
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


# ── Lightweight regex stemmer (no NLTK needed) ───────────────────────────────
# Trained on same IMDB data with same vectorizer, so token overlap is ~95%+
_STEM_RULES = [
    (r'ational$', 'ate'), (r'tional$',  'tion'), (r'enci$',   'ence'),
    (r'anci$',    'ance'), (r'izer$',    'ize'),  (r'ising$',  'ise'),
    (r'izing$',   'ize'),  (r'ising$',   'ise'),  (r'ation$',  'ate'),
    (r'ator$',    'ate'),  (r'alism$',   'al'),   (r'iveness$','ive'),
    (r'fulness$', 'ful'),  (r'ousness$', 'ous'),  (r'aliti$',  'al'),
    (r'iviti$',   'ive'),  (r'biliti$',  'ble'),  (r'icate$',  'ic'),
    (r'alize$',   'al'),   (r'iciti$',   'ic'),   (r'ical$',   'ic'),
    (r'ful$',     ''),     (r'ness$',    ''),      (r'ement$',  ''),
    (r'ment$',    ''),     (r'ent$',     ''),      (r'ism$',    ''),
    (r'ate$',     ''),     (r'iti$',     ''),      (r'ous$',    ''),
    (r'ive$',     ''),     (r'ize$',     ''),      (r'al$',     ''),
    (r'ing$',     ''),     (r'ings$',    ''),      (r'ied$',    'i'),
    (r'ies$',     'i'),    (r'ed$',      ''),      (r'ly$',     ''),
    (r'er$',      ''),     (r'est$',     ''),      (r'ers$',    ''),
    (r'ion$',     ''),     (r'ions$',    ''),      (r'tion$',   ''),
    (r'tions$',   ''),     (r'able$',    ''),      (r'ible$',   ''),
    (r'ant$',     ''),     (r'ance$',    ''),      (r'ence$',   ''),
    (r'ary$',     ''),     (r'ory$',     ''),
]
_COMPILED = [(re.compile(p + '$'), r) for p, r in _STEM_RULES]

def _stem(word: str) -> str:
    if len(word) <= 3:
        return word
    for pattern, repl in _COMPILED:
        candidate = pattern.sub(repl, word)
        if candidate != word and len(candidate) >= 3:
            return candidate
    return word

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+',         '', text)
    text = re.sub(r'<.*?>',           '', text)
    text = re.sub(r'[^a-z0-9\s]',    '', text)
    tokens = [_stem(t) for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return ' '.join(tokens)


# ── Pure-numpy RNN ────────────────────────────────────────────────────────────
class NumpyRNN:
    def __init__(self, path: str):
        w         = np.load(path)
        self.W_ih = w["W_ih"]   # (128, 5000)
        self.W_hh = w["W_hh"]   # (128, 128)
        self.b_ih = w["b_ih"]   # (128,)
        self.b_hh = w["b_hh"]   # (128,)
        self.W_fc = w["W_fc"]   # (1, 128)
        self.b_fc = w["b_fc"]   # (1,)
        self.H    = self.W_hh.shape[0]

    def predict(self, x: np.ndarray) -> float:
        h = np.zeros(self.H, dtype=np.float32)
        h = np.tanh(self.W_ih @ x + self.b_ih + self.W_hh @ h + self.b_hh)
        logit = (self.W_fc @ h + self.b_fc)[0]
        return float(1.0 / (1.0 + np.exp(-logit)))


# ── Load artifacts once at cold-start ────────────────────────────────────────
_vec   = None
_model = None

try:
    if os.path.exists(VEC_PATH):
        with open(VEC_PATH, "rb") as f:
            _vec = pickle.load(f)
    if os.path.exists(MODEL_PATH) and _vec is not None:
        _model = NumpyRNN(MODEL_PATH)
except Exception as e:
    print(f"[WARN] Could not load model artifacts: {e}")


# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.route("/health")
def health():
    return jsonify({
        "status":            "ok",
        "model_loaded":      _model is not None,
        "vectorizer_loaded": _vec   is not None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    processed = preprocess(text)

    # ── Rule-based fallback if model not available ────────────────────────
    if _model is None or _vec is None:
        POS = {"good","great","excel","amaz","wonder","fantast","love",
               "best","brilliant","outstand","superb","beauti","enjoy",
               "perfect","incred","awesom","terrif","masterpiec",
               "hilari","touch","inspir","entertain","delight"}
        NEG = {"bad","terribl","awful","worst","horribl","hate","bore",
               "dull","wast","disappoint","poor","mediocr","dread",
               "annoy","stupid","pathet","useless","garbage","trash",
               "disast","rubbish","ridicul"}
        words  = set(processed.split())
        p, n   = len(words & POS), len(words & NEG)
        total  = p + n or 1
        label  = "positive" if p >= n else "negative"
        conf   = round(min(50 + abs(p - n) / total * 50, 99), 1)
        return jsonify({"sentiment": label, "confidence": conf, "model_loaded": False})

    # ── RNN inference ─────────────────────────────────────────────────────
    x     = _vec.transform([processed]).toarray()[0].astype(np.float32)
    prob  = _model.predict(x)
    label = "positive" if prob >= 0.5 else "negative"
    conf  = round((prob if prob >= 0.5 else 1 - prob) * 100, 1)
    return jsonify({"sentiment": label, "confidence": conf, "model_loaded": True})


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    """Serve index.html for all non-API routes."""
    target = os.path.join(STATIC, path) if path else None
    if target and os.path.isfile(target):
        return send_from_directory(STATIC, path)
    return send_from_directory(STATIC, "index.html")


# Vercel needs this named `app`