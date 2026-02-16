import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from collections import Counter
from nltk.tokenize import word_tokenize
from itertools import chain
import pickle

# ------------------------------
# Load dataset
# ------------------------------
data = load_dataset("imdb")
train_texts = data["train"]["text"][5000:20000]
train_labels = torch.tensor(
    data["train"]["label"][5000:20000],
    dtype=torch.long
)

print("Label distribution:", train_labels.bincount())

# ------------------------------
# Tokenize
# ------------------------------
tokens = [word_tokenize(t.lower()) for t in train_texts]
max_len = 200

# ------------------------------
# Build vocab
# ------------------------------
tok = list(chain.from_iterable(tokens))
vocab = {"<pad>": 0, "<unk>": 1}
PAD_IDX = 0

for t in Counter(tok):
    if t not in vocab:
        vocab[t] = len(vocab)

# ------------------------------
# Numericalize
# ------------------------------
def text_to_indices(sentence):
    return [vocab.get(t, vocab["<unk>"]) for t in sentence]

encoded = [text_to_indices(s)[:max_len] for s in tokens]

padded = [
    s + [PAD_IDX] * (max_len - len(s))
    for s in encoded
]

X = torch.tensor(padded, dtype=torch.long)
y = train_labels

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ------------------------------
# Model (FIXED)
# ------------------------------
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, emb=100, hidden=64, out=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, out)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])   # ✅ CORRECT

model = SentimentRNN(len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------
# Training
# ------------------------------
for epoch in range(10):
    total_loss = 0
    correct = 0
    total = 0

    for bx, by in loader:
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == by).sum().item()
        total += by.size(0)

    acc = correct / total
    print(f"Epoch {epoch} | Loss {total_loss:.3f} | Acc {acc:.3f}")

# ------------------------------
# Save
# ------------------------------
torch.save(model.state_dict(), "sentiment.pth")

with open("vocab_sentiment.pkl", "wb") as f:
    pickle.dump(vocab, f)

with open("max_len_sentiment.pkl", "wb") as f:
    pickle.dump(max_len, f)

print("Training complete.")
