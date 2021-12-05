#!/bin/bash -x

batch_size=50
num_epochs=10

mkdir -p out

for learn_rate in 0.001 0.003 0.01 0.03 0.1 0.3
do
    for model in mlp rnn lstm
    do
        timestamp=$(date +%s)
        python model_$model.py \
            --batch_size=$batch_size --num_epochs=$num_epochs --learn_rate=$learn_rate \
            --stats_csv=out/stats.csv --stats_key=$model-$timestamp --save_model=out/$model-$timestamp.bin
    done
done

