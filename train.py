"""
train.py — trains the RNN on IMDB Dataset.csv
Produces: model.pt  and  vectorizer.pkl

Usage:
    pip install -r requirements.txt
    python train.py
"""

import pickle, re, argparse
import nltk, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from nltk.corpus   import stopwords
from nltk.stem     import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection         import train_test_split
from sklearn.preprocessing           import LabelEncoder
from torch.utils.data                import DataLoader, TensorDataset

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+",         "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"<.*?>",           "", text)
    tokens     = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    ps         = PorterStemmer()
    return " ".join(ps.stem(t) for t in tokens if t not in stop_words)


# ── Model ─────────────────────────────────────────────────────────────────────
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         default="IMDB Dataset.csv")
    parser.add_argument("--epochs",       type=int, default=10)
    parser.add_argument("--batch",        type=int, default=64)
    parser.add_argument("--max_features", type=int, default=5000)
    args = parser.parse_args()

    print("Loading data …")
    df = pd.read_csv(args.data)
    df.drop_duplicates(inplace=True)

    print("Preprocessing …  (this takes a minute)")
    df["review"] = df["review"].apply(preprocess)

    le = LabelEncoder()
    y  = le.fit_transform(df["sentiment"])   # positive=1, negative=0

    tf = TfidfVectorizer(max_features=args.max_features)
    X  = tf.fit_transform(df["review"])

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(tf, f)
    print("✓  vectorizer.pkl saved")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    X_train, X_test = X_train.toarray(), X_test.toarray()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(),
                      torch.from_numpy(y_train).float()),
        shuffle=True, batch_size=args.batch)
    test_loader  = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float(),
                      torch.from_numpy(y_test).float()),
        batch_size=args.batch)

    model     = RNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    print("Training …")
    for epoch in range(args.epochs):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(torch.sigmoid(model(Xb.unsqueeze(1)).squeeze()), yb)
            loss.backward()
            optimizer.step()
        print(f"  epoch {epoch+1}/{args.epochs}  loss={loss.item():.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            pred   = (torch.sigmoid(model(Xb.unsqueeze(1)).squeeze()) > 0.5).float()
            total  += yb.size(0)
            correct += (pred == yb).sum().item()
    print(f"Test accuracy: {correct/total*100:.2f}%")

    torch.save(model.state_dict(), "model.pt")
    print("✓  model.pt saved\n")
    print("Now run:  python app.py")


if __name__ == "__main__":
    main()
