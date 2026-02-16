import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
import random
from collections import Counter
pad_idx = 0

write = input("enter the sentence")


tokk = write.lower().split()
pd = pad_sequence(tokk, batch_first=True, padding_value = pad_idx)

print(pd)
