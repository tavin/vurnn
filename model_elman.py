import torch
from torch import nn

import model
from mytorch import GlobalMaxPool


class Elman(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, X):
        b, t, e = X.shape
        hidden = torch.zeros(b, e, dtype=X.dtype)
        output = []
        for i in range(t):
            hidden = torch.tanh(self.linear1(X[:, i, :]) + self.linear2(hidden))
            output.append(hidden[:, None, :])
        return torch.cat(output, dim=1)


class Model(model.Model):

    TITLE = 'IMDB review sentiment naive Elman model'

    def __new__(cls, *, num_classes: int, num_embeddings: int, padding_idx: int):
        return nn.Sequential(
            nn.Embedding(num_embeddings=num_embeddings, padding_idx=padding_idx, embedding_dim=cls.EMBEDDING_DIM),
            Elman(cls.EMBEDDING_DIM, cls.HIDDEN_DIM),
            nn.Linear(cls.HIDDEN_DIM, cls.HIDDEN_DIM),
            nn.ReLU(),
            GlobalMaxPool(dim=1),
            nn.Linear(cls.HIDDEN_DIM, num_classes)
        )


if __name__ == '__main__':
    model.Model = Model
    model.main()
