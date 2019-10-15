import glob
import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm
from .inference_memmap import collect_file_paths
from .dataloader import YoutubeTestDataset, DataLoader, collate_test_segments


def main():
    test_ds = YoutubeTestDataset(
        collect_file_paths(), offset=3, device="cpu",
        vocab_path="data/segment_vocabulary.csv"
    )
    vid_mapping = {}
    video_lengths = []
    global_indices = np.zeros(1704348, dtype="int8")
    global_vids = np.zeros(1704348, dtype="int32")
    for i, (video_features, segment_row, index, vid) in tqdm(enumerate(test_ds), total=1704348):
        video_lengths.append(video_features.size(0))
        global_indices[i] = index
        if vid not in vid_mapping:
            vid_mapping[vid] = len(vid_mapping)
        global_vids[i] = vid_mapping[vid]
    joblib.dump(global_indices, "data/cache/ref_indices.jl")
    joblib.dump(global_vids, "data/cache/ref_vids.jl")
    joblib.dump(vid_mapping, "data/cache/vid_mapping.jl")


if __name__ == "__main__":
    main()
