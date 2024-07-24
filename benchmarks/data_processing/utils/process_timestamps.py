import numpy as np


def txt2timestamps(timestamps_path):
    ts = None
    with open(timestamps_path, "r") as f:
        for line in f:
            values = line.strip().split(",")
            ts = [int(v) for v in values]
            break
    return ts


def paired_frameids_to_txt(data, save_path):
    with open(save_path, "w") as f:
        for ids in data:
            pair_str = [str(x) for x in ids]
            f.write(",".join(pair_str) + "\n")


def txt_to_paried_frameids(data_path):
    pairs = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            line = line.split(",")
            values = [int(x) for x in line]
            pairs.append(values)
    return pairs
