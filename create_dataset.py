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
    # print statistics
    print(f"Total groups: {len(grouped_train) + len(grouped_test)}")
    print(f"Train groups: {len(grouped_train)}")
    print(f"Test groups: {len(grouped_test)}")
    count_size_distribution = {}
    for key in set.union(set(grouped_train.keys()), set(grouped_test.keys())):
        size = key[1]
        if size not in count_size_distribution:
            count_size_distribution[size] = [0, 0]
        
        train_size = len(grouped_train[key]) if key in grouped_train else 0
        test_size = len(grouped_test[key]) if key in grouped_test else 0
        count_size_distribution[size][0] += train_size
        count_size_distribution[size][1] += test_size
        print(f"Group {key}: train size {train_size}, test size {test_size}")

    print("Size distribution of groups:")
    for size, count in count_size_distribution.items():
        print(f"Size {size}: {count} groups")

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
                f.write(json.dumps(item) + '\n')

    with open(os.path.join(output_dir, 'test_metadata.jsonl'), 'w') as f:
        for group in grouped_test.values():
            for item in group:
                f.write(json.dumps(item) + '\n')

def main():
    data_summary_path = "dataset/T2IS/data_summary.jsonl"
    n = 4
    max_resolution_threshold = 512 * 512 * n
    output_dir = f"dataset/T2IS/train_half_le_{n}" # less/equal to n 512x512 images


    with open(data_summary_path, "r") as f:
        data = [json.loads(line) for line in f]

    for d in data:
        d['size'] = (d['height'], d['width'])

    data = [d for d in data if d['height'] * d['width'] <= max_resolution_threshold]
    with open(f"dataset/T2IS/data_summary_le_{n}.jsonl", "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

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