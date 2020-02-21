import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from Model.Attention import Attention
from Model.Decoder import Decoder
from Model.Encoder import Encoder
from Model.Seq2Seq import Seq2Seq
from Utilities.Constants import *
from Utilities.NameDS import NameDataset
from Utilities.Convert import strings_to_index_tensor

BATCH_SZ = 256
EPOCH = 1000
PLOT_EVERY = 2
INPUT_DIM = len(ENCODER_INPUT)
OUTPUT_DIM = len(DECODER_INPUT)
EMBED_DIM = 5
HIDD_DIM = 512
DROPOUT = 0.5
CLIP = 1
LR = 0.005
SRC_PAD_IDX = ENCODER_INPUT['<PAD>']
TRG_PAD_IDX = DECODER_INPUT['<PAD>']

def plot_losses(loss, folder: str = "Results", filename: str = None):
    x = list(range(len(loss)))
    plt.plot(x, loss, 'b--', label="Cross Entropy Loss")
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.savefig(f"{folder}/{filename}")
    plt.close()

def run_epochs(model: Seq2Seq, iterator: DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.CrossEntropyLoss, clip: int):
    total_loss = 0
    all_losses = []
    for i in range(EPOCH):
        loss = train(model, iterator, optimizer, criterion, clip)
        total_loss += loss

        if i % PLOT_EVERY:
            all_losses.append(total_loss/PLOT_EVERY)
            plot_losses(total_loss, filename="test")

def train(model: Seq2Seq, iterator: DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.CrossEntropyLoss, clip: int):
    model.train()

    epoch_loss = 0

    for x in iterator:
        optimizer.zero_grad()

        max_len = len(max(x, key=len))
        src, src_len = strings_to_index_tensor(x, max_len, ENCODER_INPUT, SRC_PAD_IDX)
        trg, _ = strings_to_index_tensor(x, max_len, DECODER_INPUT, TRG_PAD_IDX)
        sos_tensor = torch.ones(1, len(x)).type(torch.LongTensor).to(DEVICE) * ENCODER_INPUT['<SOS>']

        output = model(src, src_len, trg, sos_tensor)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

    return loss.item()


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, src_len = batch.src
            trg = batch.trg

            output = model(src, src_len, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


df = pd.read_csv('Data/first.csv')
name_ds = NameDataset(df, "name")
dl = DataLoader(name_ds, batch_size= BATCH_SZ, shuffle=True)

attention = Attention(HIDD_DIM, HIDD_DIM)
encoder = Encoder(INPUT_DIM, EMBED_DIM, HIDD_DIM, HIDD_DIM, DROPOUT)
decoder = Decoder(OUTPUT_DIM, EMBED_DIM, HIDD_DIM, HIDD_DIM, DROPOUT, attention)

model = Seq2Seq(encoder, decoder, SRC_PAD_IDX, DEVICE).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

train(model, dl, optimizer, criterion, CLIP)
