import re
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader


def imdb_collator(pad: int) -> Callable[[Sequence], tuple[Tensor, Tensor]]:

    def collate_fn(batch):
        batch = np.array(batch, dtype=object)
        targets = batch[:, 1].astype(int)
        width = max(len(v) for v in batch[:, 0])
        examples = np.empty((len(batch), width), dtype=int)
        for x, v in zip(examples, batch[:, 0]):
            x[:len(v)] = v
            x[len(v):] = pad
        return torch.as_tensor(examples), torch.as_tensor(targets)

    return collate_fn


def imdb_loader(x: list, y: list, *, collate_fn, **kwargs) -> DataLoader:
    return DataLoader(np.array([x, y], dtype=object).transpose(),
                      collate_fn=collate_fn, **kwargs)


def arm_collator(pad: int, start: int, end: int) -> Callable[[Sequence], tuple[Tensor, Tensor]]:

    def collate_fn(batch):
        width = max(len(seq) for seq in batch) + 2
        examples = np.full((len(batch), width), pad, dtype=int)
        for seq, row in zip(batch, examples):
            row[0] = start
            row[1:len(seq)+1] = seq
            row[len(seq)+1] = end
        targets = np.c_[examples[:, 1:], np.full(len(batch), pad)]
        return torch.as_tensor(examples), torch.as_tensor(targets)

    return collate_fn


def arm_validator(dataset: str) -> Callable[[str], bool]:

    if dataset == 'ndfa':

        pattern = re.compile(r's(?:(abc!)*|(uvw!)*|(klm!)*)s')

        def validator(seq):
            return bool(pattern.fullmatch(seq))

    elif dataset == 'brackets':

        def validator(seq):
            depth = 0
            for char in seq:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth < 0:
                        return False
                else:
                    return False
            return depth == 0

    else:
        raise ValueError()

    return validator
