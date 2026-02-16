import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
import random
from collections import Counter
import string
from nltk import word_tokenize
import nltk 

document = """
Welcome to the general knowledge guide.
People often ask questions about daily life. What should I eat for breakfast? Where can I find fresh vegetables? How do I manage my time effectively? Giving clear answers is important.

Stories are a fun way to learn. Once upon a time, there was a little fox who lived in a forest. He was curious and wanted to explore new places. Every day, he met other animals and learned something new.

Traveling is exciting. You can visit mountains, beaches, cities, and villages. When you travel, it is helpful to know where the nearest hotel is, what food is popular, and which language people speak. Give yourself enough time to enjoy every destination.

Cooking is another essential skill. Recipes often ask you to add ingredients like salt, sugar, or butter. What temperature should the oven be? How long should you bake the cake? If you follow the instructions carefully, the dish turns out delicious.

Daily routines are also common topics. People ask: What time should I wake up? How can I exercise regularly? Where should I go for a morning walk? Giving small steps makes it easier to form habits.

Technology is everywhere. Smartphones, laptops, and smart devices help us communicate. When using technology, people often ask: How do I connect to Wi-Fi? What apps should I install? Give careful attention to privacy settings.

Education is important. Students often ask: What subjects are necessary? How should I study efficiently? Where can I find extra resources? Teachers guide students and give feedback on their progress.

Games and entertainment are popular too. People ask: What games are fun? Where can I watch movies? How do I play chess well? Giving tips and strategies makes games enjoyable.

Health and wellness are key. Questions like: What is a balanced diet? Where can I find a good gym? How do I stay motivated? People give advice to maintain a healthy lifestyle.

Shopping is part of daily life. What should I buy for groceries? Where can I get discounts? How do I compare products? Giving recommendations is helpful for buyers.

Nature and science spark curiosity. Why does the sun rise in the east? What causes rain? Where do birds migrate? Giving clear explanations helps people understand the world.

Art and creativity inspire people. How do I draw better? What colors should I use? Where can I exhibit my paintings? Giving constructive feedback is useful.

Social interactions matter. How should I greet someone? What questions are polite? Where should I sit in a meeting? Giving attention to manners improves relationships.

Travel stories mix curiosity and fun. What happened when I went hiking? Where did I see the most beautiful sunset? How did I learn new skills during the trip? Sharing stories encourages others to explore.

Pets and animals are beloved. What should I feed my dog? Where should I place a birdcage? How do I train a cat? Giving care tips ensures happy pets.

Festivals and events bring people together. What should I prepare for Diwali? Where is the New Year celebration happening? How can I join a local festival? Giving details helps everyone plan.

Historical facts are educational. Who was Alexander the Great? What caused the industrial revolution? Where did ancient civilizations live? Giving precise information enhances learning.

Science experiments are exciting. What chemicals react? How do I measure accurately? Where should I perform the experiment safely? Giving step-by-step guidance is essential.

Music and dance are joyful. What songs are trending? Where can I learn classical dance? How do I improve my singing? Giving encouragement improves skills.

Gardening teaches patience. What seeds should I plant? Where do I put the garden? How do I water plants correctly? Giving proper advice helps plants thrive.

Life tips are practical. What habits improve productivity? Where should I focus my energy? How do I deal with stress? Giving simple suggestions improves daily life.

Random facts and trivia. Did you know that honey never spoils? Where is the tallest building? What is the fastest animal on earth? Giving curious facts entertains readers.

Jokes and humor lighten the mood. Why did the chicken cross the road? What is the funniest movie scene? How do people laugh differently around the world? Giving a smile makes the day better.

Motivation and self-improvement. What are the goals to set? How do I stay disciplined? Where do I find inspiration? Giving positive reinforcement helps growth.

Daily conversations. Hello, how are you? What did you do today? Where are you going? Giving polite responses makes communication smooth.

Travel tips. What is the best way to pack? Where should I book tickets? How do I avoid long queues? Giving practical suggestions saves time.

Weather and seasons. What is the temperature today? Where will it rain tomorrow? How do I prepare for winter? Giving accurate information helps everyone plan.

Random knowledge expansion. What is the capital of France? Where is the Sahara Desert located? How do volcanoes erupt? Giving clear answers improves curiosity.

Shopping tips and advice. What is the best smartphone? Where can I get bargains? How do I check product quality? Giving guidance helps buyers make smart decisions.

Food and cooking experiences. What is the secret to perfect pasta? Where should I buy fresh vegetables? How do I preserve fruits? Giving detailed instructions helps cook efficiently.

Technology usage guidance. What apps improve productivity? Where do I find tutorials? How do I fix software issues? Giving recommendations enhances efficiency.

Sports and hobbies. What is the best training for running? Where can I join a club? How do I improve my skills? Giving tips motivates enthusiasts.

Storytelling exercises. What happened when I visited the mountains? Where did I meet interesting people? How do I describe scenes vividly? Giving engaging details inspires readers.

Pet care advice. What is the healthiest diet for pets? Where should I keep a fish tank? How do I train a puppy? Giving proper care keeps pets happy.

Health and fitness guidance. What exercises target the core? Where can I find healthy recipes? How do I stay consistent? Giving routines improves fitness.

Learning and skill development. What language should I learn next? Where do I find courses? How do I practice efficiently? Giving structured guidance improves learning.

Everyday life tips. What is the best way to organize a room? Where should I put furniture? How do I clean efficiently? Giving simple methods saves time.

Fun facts and trivia. What is the fastest car? Where is the largest library? How do penguins survive in cold climates? Giving interesting facts entertains readers.

This concludes the multi-topic general dataset example. The text above contains a mixture of questions, instructions, facts, stories, daily life tips, technical explanations, and casual conversation. 
It is around 12,000 words and ideal for training a next-word predictor or autocomplete model.
"""
label_text = [
    '<sos>', '<eos>'
]

tokenize = word_tokenize(document.lower())

tokens = [t for t in tokenize if t not in string.punctuation]

nltk.download('punkt')
vocab = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3}

pad_idx = 0

for t in Counter(tokens).keys():
    if t not in vocab:
        vocab[t] = len(vocab)
        
input_sentences = document.split('\n')
    
def text_to_indices(sentence,vocab):
    numerical_sentence = []
    
    for token in sentence:
        if token in vocab:
            numerical_sentence.append(vocab[token])
        else:
            numerical_sentence.append(vocab['<unk>'])
    return numerical_sentence

input_numerical_sentence =[]

for sentence in input_sentences:
    tkn = word_tokenize(sentence.lower())
    tkn = [t for t in tkn if t not in string.punctuation]
    input_numerical_sentence.append(text_to_indices(tkn, vocab))
    
training_sequence = []
    
for sentence in input_numerical_sentence:
    for i in range(len(sentence)):
        training_sequence.append(sentence[:i+1])
max_len = max(len(seq) for seq in training_sequence )
        
        
        
class CustomDataset(Dataset):
    def __init__(self, sentences, max_len):
        self.sequences = sentences
        self.max_len = max_len
        self.sos = vocab['<sos>']
        self.eos = vocab['<eos>']
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        seq = self.sequences[index]
        seq = [self.sos] + seq + [self.eos]
        padded = seq + [0]*(self.max_len - len(seq)) if len(seq) < self.max_len else seq[:self.max_len]
        x = torch.tensor(padded[:-1], dtype = torch.long)
        y = torch.tensor(padded[1:], dtype=torch.long)
        return x,y
    
dataset  = CustomDataset(training_sequence, max_len=max_len)

loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    
class Encoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            emb_size, hidden_size, num_layers, 
            bidirectional=True, dropout=dropout,batch_first=True
        )

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell   = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src):
        # src: (batch, seq_len)
        #src = src.T   # → (seq_len, batch)
        embedded = self.dropout(self.embedding(src))
        encoder_outputs, (hidden, cell) = self.rnn(embedded)
        encoder_states = encoder_outputs
        # hidden shape: (num_layers*2, batch, H)

        def bilstm(h):
            pairs = []
            for i in range(0, h.size(0), 2):
                fw = h[i]       # (1, batch, H)
                bw = h[i+1]     # (1, batch, H)
                pairs.append(torch.cat((fw, bw), dim=1))   # (1, batch, 2H)
            return torch.stack(pairs)  # (num_layers, batch, 2H) tekes collectio of items and join them along new dimensions

        hidden = self.fc_hidden(bilstm(hidden))  # → (num_layers, batch, H)
        cell   = self.fc_cell(bilstm(cell))      # → (num_layers, batch, H)

        return encoder_states, hidden, cell

    
class Decoder(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size, num_layers, dropout, num_heads=4, encoder_hidden_size=None):
        super().__init__()
        self.embedding = nn.Embedding(output_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.d_head = hidden_size // num_heads
        assert hidden_size % num_heads == 0
        self.rnn = nn.LSTM(emb_size + hidden_size, hidden_size, num_layers, dropout=dropout)
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, hidden, cell, prev_states):
        x = x.unsqueeze(0)
        embedded = self.dropout(self.embedding(x))

        if len(prev_states) == 0:
            batch = embedded.size(1)
            context = torch.zeros(1, batch, hidden.size(2), device=x.device)
        else:
            H_prev = torch.cat(prev_states, dim=0)  # (T_prev, batch, enc_hidden)
            

            h_t = hidden[-1].unsqueeze(0)
            Q = self.W_q(h_t)
            K = self.W_k(H_prev)
            V = self.W_v(H_prev)

            # ... rest of multi-head attention remains the same


            # ---- 2) Split into heads ----
            def split_heads(x):
                # x: (seq_len, batch, H)
                seq_len, batch, H = x.size()
                x = x.view(seq_len, batch, self.num_heads, self.d_head)
                x = x.permute(1, 2, 0, 3)  # (batch, heads, seq_len, d_head)
                return x

            Q = split_heads(Q)      # (batch, heads, 1, d_head)
            K = split_heads(K)      # (batch, heads, T_prev, d_head)
            V = split_heads(V)      # (batch, heads, T_prev, d_head)

            # ---- 3) Compute scaled dot-product attention ----
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)  # (batch, heads, 1, T_prev)
            weights = self.softmax(scores)  # (batch, heads, 1, T_prev)
            context = torch.matmul(weights, V)  # (batch, heads, 1, d_head)

            # ---- 4) Concatenate heads ----
            context = context.permute(2, 0, 1, 3).contiguous()  # (1, batch, heads, d_head)
            context = context.view(1, context.size(1), self.num_heads * self.d_head)  # (1, batch, H)


        # ---- 5) LSTM input ----
        rnn_input = torch.cat((embedded, context), dim=2)  # (1, batch, E+H)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc(output.squeeze(0))  # (batch, vocab)
        return prediction, hidden, cell






class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = len(vocab)

        outputs = torch.zeros(trg_len, batch_size, vocab_size)

        encoder_states, hidden, cell = self.encoder(src)

        x = trg[:, 0]   # first token (<sos>)
        prev_outputs = []
        for t in range(1, trg_len):
            pred, hidden, cell = self.decoder(x, hidden, cell,prev_outputs)
            prev_outputs.append(hidden[-1:].detach())  # last layer hidden
  #detach it to prevent this step from back propagating
            outputs[t] = pred
            x = pred.argmax(1)   # greedy decoding

        return outputs


input_size = len(vocab)
output_size = len(vocab)
hidden_size = 128
n_heads = 4
num_layers = 5
dropout = 0.5
emb_size = 50
encoder = Encoder(input_size, emb_size, hidden_size, num_layers, dropout) 
decoder = Decoder(output_size, emb_size, hidden_size,num_layers ,dropout,n_heads)   
model = Seq2Seq(encoder,decoder)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=0.001)  

for epoch in range(5):
    for src, trg in loader:
        optimizer.zero_grad()
        
        outputs = model(src, trg)
        
        outputs = outputs[1:].reshape(-1,output_size)
        
        trg = trg[:, 1:].reshape(-1)

        
        loss = criterion(outputs, trg)
        
        loss.backward()
        
        optimizer.step()
        
    print(f"Epoch {epoch+1} | Loss = {loss.item():.4f}")

print("\nTraining complete.")
        

index_to_word = {v: k for k, v in vocab.items()}


def predict(model,sentence):
    model.eval()
    
    tokens = sentence.lower().split()
    ids = text_to_indices(tokens,vocab)
    
    src = torch.tensor(ids)
    src = src.unsqueeze(0)
    if src.size(1) < max_len:
        src = torch.nn.functional.pad(
            src, (0, max_len - src.size(1)), value=pad_idx
        )
    with torch.no_grad():
        output, hidden, cell = model.encoder(src)
        
    x = torch.tensor([vocab['<sos>']])
    
    outputs = []
    prev_outputs = []
    for _ in range(max_len):
        with torch.no_grad():
            out, hidden, cell = model.decoder(x, hidden, cell, prev_outputs)
            
        next_tok = out.argmax(1).item()
        prev_outputs.append(hidden[-1:].detach())
        if next_tok== vocab['<eos>']:
            break
        outputs.append(next_tok)
        x = torch.tensor([next_tok])
        
    words = [index_to_word[i] for i in outputs]
    
    return words[-1]


write = input("wnter the words")


print("the predicteds word is")

print(predict(model,write))
        
        
    
    
        
#i havent trained this properly and epoch is very less so it is not accurate enough
        
        
        
        
        
        
        
        
