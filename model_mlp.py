from torch import nn

import model
from mytorch import GlobalMaxPool


class Model(model.Model):

    KEY = 'mlp'
    TITLE = 'IMDB review sentiment baseline MLP model'

    def __new__(cls, *, num_classes: int, num_embeddings: int, padding_idx: int):
        return nn.Sequential(
            nn.Embedding(num_embeddings=num_embeddings, padding_idx=padding_idx, embedding_dim=cls.EMBEDDING_DIM),
            nn.Linear(cls.EMBEDDING_DIM, cls.HIDDEN_DIM),
            nn.ReLU(),
            GlobalMaxPool(dim=1),
            nn.Linear(cls.HIDDEN_DIM, num_classes)
        )


if __name__ == '__main__':
    model.Model = Model
    model.main()
