# ======================================================================
#   SEQ2SEQ SENTIMENT ANALYSIS WITH IMDB (NO ATTENTION)
#   Encoder → LSTM → Decoder → predict "positive" or "negative" sentence
# ======================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
import random
from collections import Counter
# -----------------------------------------------------
# Load IMDB dataset (HuggingFace)
# -----------------------------------------------------
data = load_dataset("imdb")

train_texts = data["train"]["text"][5000:20000]  # limit for safety
train_labels = data["train"]["label"][5000:20000]

# Sentiment labels (converted to sequences)
# 0 → "<sos> negative <eos>"
# 1 → "<sos> positive <eos>"
label_text = {
    0: ["<sos>", "negative", "<eos>"],
    1: ["<sos>", "positive", "<eos>"]
}

# -----------------------------------------------------
# Tokenizer  
# simple lowercase + split
# -----------------------------------------------------
def tokenize(text):
    return text.lower().split()


tokenized_reviews = [tokenize(t) for t in train_texts]

tok = [t for sentence in tokenized_reviews for t in sentence]

vocab={"<pad>":0, "<unk>":1, "<sos>":2, "<eos>":3, "negative":4, "positive":5}
pad_idx = 0

for t in Counter(tok).keys():
    if t not in vocab:
        vocab[t]= len(vocab)
    
def text_to_indices(text):
    return [vocab.get(t,vocab["<unk>"]) for t in text]

# -----------------------------------------------------
# Encode text
# - truncate to max length
# - convert tokens to IDs
# -----------------------------------------------------
max_len = 10

def encode(tokens):
    tokens = tokens[:max_len]
    ids = text_to_indices(tokens)
    return torch.tensor(ids, dtype=torch.long)


encoded_reviews = [encode(x) for x in tokenized_reviews]

# -----------------------------------------------------
# Pad sequences (right-side padding)
# pad_sequence returns batches shaped (batch, max_len)
# Right side padding = all padding tokens go to the end
# e.g. [2,5,6] → [2,5,6,<pad>,<pad>,...]
# -----------------------------------------------------
padded_reviews = pad_sequence(encoded_reviews, batch_first=True,
                              padding_value=pad_idx)

# Ensure fixed length (if shorter)
if padded_reviews.size(1) < max_len:
    padded_reviews = torch.nn.functional.pad(
        padded_reviews,
        (0, max_len - padded_reviews.size(1)),
        value=pad_idx
    )

# -----------------------------------------------------
# Encode label sequences for seq2seq training
# -----------------------------------------------------

encoded_labels = [
    torch.tensor([vocab.get(tok, vocab["<unk>"]) for tok in label_text[l]], dtype=torch.long) #train labels ma value 0 ra 1 matra hunchan
    for l in train_labels
]




class SentDataset(Dataset):
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]


dataset = SentDataset(padded_reviews, encoded_labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ======================================================
#                      ENCODER
# ======================================================
class Encoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layers,bidirectional=True, dropout=dropout)
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, src):
        # src shape: (batch, seq_len)
        src = src.T   # convert to (seq_len, batch) for LSTM. yo nagarera batch_first = true pani garna milcha
        embedded = self.dropout(self.embedding(src))
        encoder_state, (hidden, cell) = self.rnn(embedded)
        #encoder_state → (seq_len, batch, hidden_size*2) (concatenated forward+backward), hidden → (num_layers*2, batch, hidden_size)
        #since shape of hidden = (num_layers*2, batch, hidden_size), hidden[0] is first layer forward hidden size, hidden[1] is first layer backward hidden hise and we concatenate that\
        def bilstm(h):
            pairs = []
            for i in range(0, h.size(0)):
                pair = torch.cat((h[i:i+1], h[i+1:i+2]),dim = 2)
                pairs.append(pair)
                return torch.cat(pairs,dim = 0)
        hidden = self.fc_hidden(bilstm(hidden)) #Shape before concat: (1, batch, H) + (1, batch, H), Shape after concat: (1, batch, 2*H)
        cell = self.fc_cell(bilstm(cell))
        
        return encoder_state,hidden, cell


# ======================================================
#                      DECODER
# ======================================================
class Decoder(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(hidden_size*2+emb_size, hidden_size, num_layers, dropout=dropout) #hidden_size*2 → context vector from attention over encoder states, emb_size → embedded previous token
        
        self.energy = nn.Linear(hidden_size*3,1)
        
        self.softmax = nn.Softmax(dim = 0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_states,hidden, cell):
        # x shape: (batch,)
        x = x.unsqueeze(0)  # LSTM expects (1, batch)
        embedded = self.dropout(self.embedding(x))
        #encoder_states shape = (seq_len, batch, hidden_size*2), hidden shape = (n_layers, batch, hidden_size)
        
        #Repeat hidden along seq_len dimension to align with encoder states, h_reshaped.shape = (seq_len, batch, hidden_size)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length,1,1 )
        
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim = 2)))
        
        attention = self.softmax(energy)
        #(seq_len, N , 1)
        
        attention = attention.permute(1,2,0)
        
        encoder_states = encoder_states.permute(1,0,2)
        
        context_vector = torch.bmm(attention, encoder_states).permute(1,0,2) #(N,1,hidden_size*2) (it it batch matric multiplication for ci)
        
        rnn_input = torch.cat((context_vector, embedded), dim=2)

        # output shape: (1, batch, hidden)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # predictions shape: (batch, vocab_size)
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden, cell


# ======================================================
#                      SEQ2SEQ
# ======================================================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_force_ratio=0.5):

        batch_size = src.size(0)
        trg_len = trg.size(1)       # number of tokens in label sentence
        vocab_size = len(vocab)

        # To store outputs
        outputs = torch.zeros(trg_len, batch_size, vocab_size)

        encoder_states,hidden, cell = self.encoder(src)

        # First decoder input = <sos>
        x = trg[:, 0]      # first token for each batch

        for t in range(1, trg_len):

            output, hidden, cell = self.decoder(x,encoder_states, hidden, cell)

            # Store prediction for this timestep
            outputs[t] = output

            # Choose top prediction
            best_guess = output.argmax(1)

            # teacher forcing: decide if using true next token
            x = trg[:, t] if random.random() < teacher_force_ratio else best_guess

        return outputs


# -----------------------------------------------------
# Initialize model
# -----------------------------------------------------
input_size = len(vocab)
output_size = len(vocab)
emb_size = 25
hidden_size = 5
num_layers = 1
dropout = 0.5

encoder = Encoder(input_size, emb_size, hidden_size, num_layers, dropout)
decoder = Decoder(output_size, emb_size, hidden_size, num_layers, dropout)
model = Seq2Seq(encoder, decoder)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ======================================================
# Training Loop
# ======================================================
for epoch in range(1):
    for src, trg in loader:

        optimizer.zero_grad()

        # pad trg to equal length
        trg = pad_sequence(trg, batch_first=True, padding_value=pad_idx)

        outputs = model(src, trg)

        # outputs: (trg_len, batch, vocab)
        # trg:     (batch, trg_len)
        outputs = outputs[1:].reshape(-1, output_size)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(outputs, trg)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} | Loss = {loss.item():.4f}")

print("\nTraining complete.")


def predict(model, sentence):
    model.eval()

    # 1. Tokenize
    tokens = sentence.lower().split()
    ids = text_to_indices(tokens)
    
    # 2. Convert to tensor
    src = torch.tensor([ids])
    
    # 3. Pad to max_len
    if src.size(1) < max_len:
        src = torch.nn.functional.pad(
            src, (0, max_len - src.size(1)), value=pad_idx
        )

    # 4. Encode input → hidden, cell
    with torch.no_grad():
       hello, hidden, cell = model.encoder(src)

    # 5. Start decoder with <sos>
    x = torch.tensor([vocab["<sos>"]])

    outputs = []

    # 6. Decode up to 5 tokens
    for _ in range(5):
        with torch.no_grad():
            out, hidden, cell = model.decoder(x, hello , hidden, cell)

        next_tok = out.argmax(1).item()
        outputs.append(next_tok)

        x = torch.tensor([next_tok])

        if next_tok == vocab["<eos>"]:
            break

    # Convert IDs back to words
    inv_vocab = {v: k for k, v in vocab.items()}
    return [inv_vocab[i] for i in outputs]



user_input = input("Enter sentence: ")
print(predict(model, user_input)[0])









