import os
import json
import re
from itertools import chain, groupby
import pandas as pd
import numpy as np
import random


def split_train_test_from_metadata(
        data : list[dict],
        train_ratio=0.5,
        group_keys = ['uid', 'size'],
        seed=42
        ):

    random.seed(seed)
    random.shuffle(data)

    train_set = pd.DataFrame(data[:int(len(data)*train_ratio)])
    test_set = pd.DataFrame(data[int(len(data)*train_ratio):])

    grouped_train = train_set.groupby(group_keys)
    grouped_test = test_set.groupby(group_keys)

    grouped_train = {
        name: group.to_dict('records')
        for name, group in grouped_train
    }
    grouped_test = {
        name: group.to_dict('records')
        for name, group in grouped_test
    }

    return grouped_train, grouped_test


def save_split_dataset(
        grouped_train,
        grouped_test,
        output_dir
        ):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'train_metadata.jsonl'), 'w') as f:
        for group in grouped_train.values():
            for item in group:
                f.write(json.dumps(item) + '\n')s

    with open(os.path.join(output_dir, 'test_metadata.jsonl'), 'w') as f:
        for group in grouped_test.values():
            for item in group:
                f.write(json.dumps(item) + '\n')

def main():
    data_summary_path = "dataset/T2IS/data_summary.jsonl"
    output_dir = "dataset/T2IS/train_half"

    with open(data_summary_path, "r") as f:
        data = [json.loads(line) for line in f]

    for d in data:
        d['size'] = (d['height'], d['width'])

    grouped_train, grouped_test = split_train_test_from_metadata(
        data,
        train_ratio=0.5,
        group_keys=['uid', 'size'],
        seed=42
    )

    save_split_dataset(
        grouped_train,
        grouped_test,
        output_dir
    )

if __name__ == "__main__":
    main()