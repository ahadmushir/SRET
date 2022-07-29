import torch
from torch import nn
import sys

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

import torch
import pandas as pd
from collections import Counter

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
      file1 = open('template1_train.txt', 'r')
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

to_gen = str(sys.argv[1])
dataset = Dataset(sequence_length=4)
model = Model(dataset)

model.load_state_dict(torch.load("/work/mushir/lstm/lstm_saved"))
print(predict(dataset, model, text=to_gen))
