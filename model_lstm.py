from torch import nn

import model
from mytorch import GlobalMaxPool


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, X):
        return self.lstm(X)[0]


class Model(model.Model):

    KEY = 'lstm'
    TITLE = 'IMDB review sentiment pytorch LSTM model'

    def __new__(cls, *, num_classes: int, num_embeddings: int, padding_idx: int):
        return nn.Sequential(
            nn.Embedding(num_embeddings=num_embeddings, padding_idx=padding_idx, embedding_dim=cls.EMBEDDING_DIM),
            LSTM(cls.EMBEDDING_DIM, cls.HIDDEN_DIM),
            nn.Linear(cls.HIDDEN_DIM, cls.HIDDEN_DIM),
            nn.ReLU(),
            GlobalMaxPool(dim=1),
            nn.Linear(cls.HIDDEN_DIM, num_classes)
        )


if __name__ == '__main__':
    model.Model = Model
    model.main()
