import torch


class GlobalMaxPool(torch.nn.Module):

    """
    Reduce a tensor over a single dimension by taking the 1d maximum.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.__dim = dim

    def forward(self, X):
        return torch.amax(X, self.__dim)
