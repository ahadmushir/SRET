import torch
from torch import nn
import pandas as pd
from collections import Counter
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequence_length):
      # self.args = args
      self.sequence_length = sequence_length
      self.words = self.load_words()
      self.uniq_words = self.get_uniq_words()

      self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
      self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

      self.words_indexes = [self.word_to_index[w] for w in self.words]
    # TODO: Change the custom functions as it can be much cleaner and straight-forward
    def load_words(self):
      file1 = open('template2_train.txt', 'r')
      Lines_anno_t = file1.readlines()
      tt = " ".join(Lines_anno_t)
      return tt.split(" ")
      # train_df = pd.read_csv('data/reddit-cleanjokes.csv')
      # text = train_df['Joke'].str.cat(sep=' ')
      # return text.split(' ')

    def get_uniq_words(self):
      word_counts = Counter(self.words)
      return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
      return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, index):
      return (
          torch.tensor(self.words_indexes[index:index+self.sequence_length]),
          torch.tensor(self.words_indexes[index+1:index+self.sequence_length+1]),
      )

def train(dataset, model, sequence_length, batch_size, max_epochs):
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(max_epochs):
        state_h, state_c = model.init_state(sequence_length)

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })

# Prediction
def predict(dataset, model, text, next_words=100):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words

dataset = Dataset(sequence_length=4)
model = Model(dataset)
train(dataset, model, max_epochs=3, batch_size=32, sequence_length=4)

print(predict(dataset, model, text='<annotation_start_m> define a function cache <annotation_end_m>#<method_sign_start>'))

torch.save(model.state_dict(), './lstm_saved_template2')
