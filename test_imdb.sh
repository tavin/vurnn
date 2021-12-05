#!/bin/bash -x

batch_size=50
num_epochs=5

model=${1?}
learn_rate=${2?}

test -f "$model" || exit
key=$(basename "$model" .bin)

mkdir -p test
python "model_${key%-*}.py" \
    --batch_size=$batch_size --num_epochs=$num_epochs --learn_rate=$learn_rate \
    --stats_csv=test/stats.csv --stats_key="$key" --load_model="$model" --test_split

