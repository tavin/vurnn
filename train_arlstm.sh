#!/bin/bash -x

batch_size=50
num_epochs=10

for learn_rate in 0.01 0.03 0.1 0.3 1 3
do
    for dataset in brackets ndfa
    do
        python arlstm.py --dataset=$dataset --batch_size=$batch_size --num_epochs=$num_epochs --learn_rate=$learn_rate
    done
done

