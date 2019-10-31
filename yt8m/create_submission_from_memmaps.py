import gc
import glob
import random
import argparse
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import joblib

from .telegram_tokens import BOT_TOKEN, CHAT_ID
from .telegram_sender import telegram_sender

BATCH_SIZE = 1000


def inverse_label_mapping(vocab_path="./data/segment_vocabulary.csv"):
    vocab = pd.read_csv(vocab_path)
    return {
        index: label for label, index in zip(vocab["Index"], vocab.index)
    }


@telegram_sender(token=BOT_TOKEN, chat_id=CHAT_ID, name="Creating submission")
def main():
    PRUNING_FREQUENCY = 100000
    assert PRUNING_FREQUENCY % BATCH_SIZE == 0
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model-dir', type=str, default="data/cache/predictions/")
    arg('--model-names', nargs="+")
    args = parser.parse_args()

    Path("data/cache/inference").mkdir(exist_ok=True, parents=True)

    model_dir = Path(args.model_dir)
    predictions = []
    with open(f"data/cache/inference/{datetime.now().strftime('log_%m%d_%H%M')}.txt", "w") as fout:
        if args.model_names is None:
            for filepath in glob.glob(str(model_dir / "*.npy")):
                print(filepath)
                fout.write(Path(filepath).stem + "\n")
                predictions.append(np.memmap(
                    filepath,
                    "int16", mode="r", shape=(1704348, 1000)
                ))
        else:
            for model_name in args.model_names:
                print(str(model_dir / model_name))
                fout.write((model_dir / model_name).stem + "\n")
                predictions.append(np.memmap(
                    str(model_dir / model_name),
                    "int16", mode="r", shape=(1704348, 1000)
                ))
        fout.write(f"Total: {len(predictions)} models\n")
    n_bins = 10000
    buckets = [[[] for _ in range(n_bins)] for _ in range(1000)]
    minimums = np.array([-1] * 1000)
    # not_filled = np.array([True] * 1000)
    indices = joblib.load("data/cache/ref_indices.jl")
    vids = joblib.load("data/cache/ref_vids.jl")
    vid_mapping = joblib.load("data/cache/vid_mapping.jl")
    with torch.no_grad():
        for cnt in tqdm(range(0, predictions[0].shape[0], BATCH_SIZE)):
            # shape(frames / 5, n_classes)
            batch = np.round(np.mean([
                x[cnt:cnt+BATCH_SIZE] for x in predictions
            ], axis=0), 0).astype("int")
            # classes = np.where(row > cutoff)[0]
            row_ids, classes = np.where(
                # not_filled | (row >= minimums)
                batch >= minimums[None, :]
            )
            for row_id, class_idx in zip(row_ids, classes):
                buckets[class_idx][batch[row_id, class_idx]].append(
                    vids[cnt + row_id] * 1000 + indices[cnt + row_id])
                # minimums[class_idx] = min(
                #     minimums[class_idx], bucket_idx)

            if (cnt + BATCH_SIZE) % PRUNING_FREQUENCY == 0:
                print("Pruning")
                counts = np.array([0] * 1000)
                for class_idx in range(1000):
                    counts[class_idx] = 0
                    new_minimum = -1
                    for i in range(n_bins - 1, -1, -1):
                        if i < minimums[class_idx]:
                            # No point moving further
                            # minimums[class_idx] == n_bins means
                            #   no entries has been appended yet
                            break
                        if counts[class_idx] >= 100000:
                            buckets[class_idx][i] = []
                            continue
                        counts[class_idx] += len(buckets[class_idx][i])
                        if counts[class_idx] >= 100000:
                            if new_minimum == -1:
                                new_minimum = i
                    # Make sure either it hasn't accumulated enough
                    #   or we get the minimum right
                    assert counts[class_idx] <= 100000 or new_minimum != n_bins
                    minimums[class_idx] = new_minimum
                print("=" * 10)
                print(pd.Series(counts).describe())
                print("-" * 10)
                print(pd.Series(minimums).describe())
                print("=" * 10)
                # not_filled = (counts < 100000)
                gc.collect()

    # Video names
    # (Only feasible for Python 3.7+, where dict objects are ordered.)
    vid_names = list(vid_mapping.keys())
    del vid_mapping
    gc.collect()

    class_predictions = []
    for class_idx in range(1000):
        tmp = []
        for i in range(n_bins - 1, -1, -1):
            if len(tmp) + len(buckets[class_idx][i]) >= 100000:
                tmp += random.choices(
                    buckets[class_idx][i],
                    k=100000 - len(tmp)
                )
                break
            # shuffle the bucket
            random.shuffle(buckets[class_idx][i])
            tmp += buckets[class_idx][i]
        class_predictions.append(" ".join([
            f"{vid_names[(encoded // 1000)]}:{(encoded % 1000)*5}" for encoded in tmp
        ]))

    mapping = inverse_label_mapping()
    df_sub = pd.DataFrame(
        [(
            mapping[i],
            entries
        ) for i, entries in enumerate(class_predictions)],
        columns=["Class", "Segments"]
    )
    df_sub.to_csv("sub.csv", index=False)


if __name__ == "__main__":
    main()
