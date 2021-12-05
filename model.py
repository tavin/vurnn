import csv
import io
import time
from typing import Callable

import click
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_rnn import load_imdb
from mydata import imdb_collator, imdb_loader


class Model:

    # KEY = 'abc'
    # TITLE = 'IMDB ...'

    EMBEDDING_DIM = 300
    HIDDEN_DIM = 300

    LOSS_FUNCTION = nn.functional.cross_entropy

    def __new__(cls, *, num_classes: int, num_embeddings: int, padding_idx: int):
        raise NotImplementedError()


def train(data: DataLoader, model: nn.Module, objective: Callable, optimizer: optim.Optimizer, t0: float) -> float:

    cum_loss: float = 0
    batches: int = 0
    progress: int = 0

    def log_progress():
        print(f'Cumulative loss: {cum_loss:.1e}\t[{batches}/{len(data)}]')

    for batches, (examples, targets) in enumerate(data):
        if progress < (progress := (time.monotonic() - t0) // 30.0):
            log_progress()
        logits = model(examples)
        loss = objective(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_loss += loss.item()

    batches += 1
    log_progress()
    return cum_loss


@torch.inference_mode()
def validate(data: DataLoader, model: nn.Module) -> float:

    correct: int = 0
    total: int = 0

    for examples, targets in data:
        logits = model(examples)
        predictions = logits.argmax(dim=1)
        correct += torch.sum(predictions == targets).item()
        total += len(targets)

    acc = correct / total
    print(f'Validation accuracy: {correct}/{total}={acc:.3f}')
    return acc


@click.command()
@click.option('--batch_size', required=True, type=int)
@click.option('--learn_rate', required=True, type=float)
@click.option('--num_epochs', default=1, type=int)
@click.option('--load_model', type=click.File('rb', lazy=True))
@click.option('--save_model', type=click.File('wb', lazy=True))
@click.option('--stats_csv', type=click.File('a', lazy=False))
@click.option('--stats_key', type=str)
@click.option('--test_split', is_flag=True)
def main(
    batch_size: int,
    learn_rate: float,
    num_epochs: int,
    load_model: io.RawIOBase,
    save_model: io.RawIOBase,
    stats_csv: io.TextIOBase,
    stats_key: str,
    test_split: bool,
):

    def log_stats(_stats):
        pass

    if stats_csv:
        log_stats = csv.writer(stats_csv).writerow
        if not stats_key:
            stats_key = Model.KEY

    (x_train, y_train), (x_val, y_val), (i2w, w2i), num_classes = load_imdb(final=test_split)
    words = len(i2w)
    padding = w2i['.pad']
    collate_fn = imdb_collator(padding)
    xy_train = imdb_loader(x_train, y_train, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    xy_val = imdb_loader(x_val, y_val, collate_fn=collate_fn, batch_size=batch_size)
    model = Model(num_classes=num_classes, num_embeddings=words, padding_idx=padding)
    objective = Model.LOSS_FUNCTION
    optimizer = optim.SGD(model.parameters(), lr=learn_rate)

    if load_model:
        # didn't save optimizer state but oh well it's just vanilla SGD
        model.load_state_dict(torch.load(load_model))

    print(f'# {Model.TITLE}')
    print(f'Number of epochs: {num_epochs}')
    print(f'Training + validation split: {len(y_train)} + {len(y_val)}')
    print(f'Batch size: {batch_size}')
    print(f'Optimizer: {optimizer}')
    print()

    for epoch in range(1, num_epochs + 1):
        print(f'# Epoch {epoch:02d}')
        t0 = time.monotonic()
        model.train()
        loss = train(xy_train, model, objective, optimizer, t0)
        model.eval()
        acc = validate(xy_val, model)
        elapsed = time.monotonic() - t0
        print(f'Elapsed time: {elapsed:.1f}s\n')
        log_stats([stats_key, batch_size, learn_rate, epoch, elapsed, loss, acc])

    if save_model:
        torch.save(model.state_dict(), save_model)


if __name__ == '__main__':
    raise NotImplementedError()
