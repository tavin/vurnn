import random
import re
import time

import click
import numpy as np
import torch
from torch import nn, optim, distributions as dist
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_rnn import load_brackets, load_ndfa
from mydata import arm_collator, arm_validator


FIVE_TEMPERATURES = np.linspace(0., 1., num=5)

RESET = '\033[0m'
GREY = '\033[90m'
RED = '\033[91m'
GREEN = '\033[92m'


class LSTM(nn.Module):

    """
    Wrap nn.LSTM in order to discard unwanted outputs, and set batch_first=True while we're at it.
    """

    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, **kwargs)

    def forward(self, X):
        return self.lstm(X)[0]


class Model:

    EMBEDDING_DIM = 32
    HIDDEN_DIM = 16

    def __new__(cls, *, num_chars: int, padding_idx: int):

        model = nn.Sequential(
            nn.Embedding(num_embeddings=num_chars, padding_idx=padding_idx, embedding_dim=cls.EMBEDDING_DIM),
            LSTM(cls.EMBEDDING_DIM, cls.HIDDEN_DIM),
            nn.Linear(cls.HIDDEN_DIM, num_chars),
        )

        def objective(logits, targets):
            # pytorch cross_entropy expects class probabilities to come between the batch and other dimensions
            return cross_entropy(logits.movedim(-1, 1), targets, ignore_index=padding_idx)

        return model, objective


@click.command()
@click.option('--batch_size', required=True, type=int)
@click.option('--dataset', required=True, type=click.Choice(['ndfa', 'brackets']))
@click.option('--learn_rate', required=True, type=float)
@click.option('--num_epochs', default=1, type=int)
@click.option('--validation_size', default=100, type=int)
def main(
    batch_size: int,
    dataset: str,
    learn_rate: float,
    num_epochs: int,
    validation_size: int,
):

    x_train, (i2w, w2i) = dict(ndfa=load_ndfa, brackets=load_brackets)[dataset](n=150_000)
    num_chars = len(i2w)
    padding = w2i['.pad']
    collate_fn = arm_collator(padding, w2i['.start'], w2i['.end'])
    data = DataLoader(x_train, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    model, objective = Model(num_chars=num_chars, padding_idx=padding)
    optimizer = optim.SGD(model.parameters(), lr=learn_rate)
    validator = arm_validator(dataset)
    board = SummaryWriter(comment=f'_{dataset}')

    print(f'# ARLSTM | {dataset}')
    print(f'i2w: {i2w}')
    print(f'w2i: {w2i}')
    print(f'Number of epochs: {num_epochs}')
    print(f'Batch size: {batch_size}')
    print(f'Optimizer: {optimizer}')
    print()

    def pretty(seq):
        return re.sub(r' (?=\.end$)', f' {GREY}', ' '.join(i2w[i] for i in seq[1:]))

    def train():
        cum_loss: float = 0
        for examples, targets in data:
            logits = model(examples)
            loss = objective(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
        print(f'Cumulative loss: {cum_loss:.1e}')
        return cum_loss

    def validate_synthesis(temperature):
        wrong = 0
        correct = 0
        for _ in range(validation_size):
            seq = [w2i['.start']]
            while seq[-1] != w2i['.end'] and len(seq) < 100:
                logits = model(torch.as_tensor(seq).reshape(1, -1))[0, -1]
                token = dist.Categorical(logits=logits/temperature).sample() if temperature > 0. else logits.argmax()
                seq.append(token)
            if seq[-1] == w2i['.end'] and validator(''.join(i2w[i] for i in seq[1:-1])):
                correct += 1
                if correct == 1:
                    print(GREEN + pretty(seq) + RESET)
            else:
                wrong += 1
                if wrong == 1:
                    print(RED + pretty(seq) + RESET)
        accuracy = correct / validation_size
        print(f'Synthesis accuracy: {accuracy:.3f}; temperature: {temperature}')
        return accuracy

    def validate_completion(temperature):
        wrong = 0
        correct = 0
        for seq in random.sample(x_train, validation_size):
            seq = [w2i['.start']] + seq[0:len(seq)//2]
            while seq[-1] != w2i['.end'] and len(seq) < 100:
                logits = model(torch.as_tensor(seq).reshape(1, -1))[0, -1]
                token = dist.Categorical(logits=logits/temperature).sample() if temperature > 0. else logits.argmax()
                seq.append(token)
            if seq[-1] == w2i['.end'] and validator(''.join(i2w[i] for i in seq[1:-1])):
                correct += 1
                if correct == 1:
                    print(GREEN + pretty(seq) + RESET)
            else:
                wrong += 1
                if wrong == 1:
                    print(RED + pretty(seq) + RESET)
        accuracy = correct / validation_size
        print(f'Completion accuracy: {accuracy:.3f}; temperature: {temperature}')
        return accuracy

    for epoch in range(1, num_epochs + 1):
        print(f'# Epoch {epoch:02d}')
        t0 = time.monotonic()
        with torch.enable_grad():
            model.train()
            train_loss = train()
            board.add_scalar('Loss/Train', train_loss, epoch)
        with torch.inference_mode():
            model.eval()
            val_acc1 = {str(temperature): validate_synthesis(temperature) for temperature in FIVE_TEMPERATURES}
            val_acc2 = {str(temperature): validate_completion(temperature) for temperature in FIVE_TEMPERATURES}
            board.add_scalars('Accuracy/Synthesis', val_acc1, epoch)
            board.add_scalars('Accuracy/Completion', val_acc2, epoch)
        elapsed = time.monotonic() - t0
        print(f'Elapsed time: {elapsed:.1f}s\n')

    board.add_hparams(
        {'dataset': dataset, 'learn': learn_rate, 'batch': batch_size},
        {'hparam/loss': train_loss, 'hparam/accuracy': np.median([*val_acc1.values(), *val_acc2.values()])}
    )

    board.close()


if __name__ == '__main__':
    main()
