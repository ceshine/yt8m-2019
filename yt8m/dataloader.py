import os
import logging

import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

LOGGER = logging.getLogger("dataset")


def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    """Dequantize the feature from the byte format to the float format.

    Args:
      feat_vector: the input 1-d vector.
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      A float vector which has the same shape as feat_vector.
    """
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias


class YoutubeSegmentDataset(IterableDataset):
    def __init__(self, file_paths, seed=939, debug=False,
                 vocab_path="./data/segment_vocabulary.csv",
                 epochs=1, max_examples=None, offset=0):
        super(YoutubeSegmentDataset).__init__()
        print("Offset:", offset)
        self.file_paths = file_paths
        self.seed = seed
        self.debug = debug
        self.max_examples = max_examples
        vocab = pd.read_csv(vocab_path)
        self.label_mapping = {
            label: index for label, index in zip(vocab["Index"], vocab.index)
        }
        self.epochs = epochs
        self.offset = offset

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            seed = self.seed
        else:  # in a worker process
               # split workload
            if worker_info.num_workers > 1 and self.epochs == 1:
                raise ValueError("Validation cannot have num_workers > 1!")
            seed = self.seed + worker_info.id
        return self.generator(seed)

    def prepare_one_sample(self, row):
        example = tf.train.SequenceExample()
        tmp = example.FromString(row.numpy())
        context, video_features = tmp.context, tmp.feature_lists

        vid_labels = list(context.feature['labels'].int64_list.value)
        vid_labels_encoded = set([
            self.label_mapping[x] for x in vid_labels if x in self.label_mapping
        ])
        segment_labels = np.array(
            context.feature['segment_labels'].int64_list.value).astype("int64")
        segment_start_times = np.array(
            context.feature['segment_start_times'].int64_list.value)
        segment_scores = np.array(
            context.feature['segment_scores'].float_list.value).astype("int64")
        vid = context.feature['id'].bytes_list.value[0].decode('utf8')

        # Transform label
        segment_labels = np.array([
            self.label_mapping[x] for x in segment_labels])

        # Negative Mining: Shape (1000)
        if self.debug:
            if not vid_labels_encoded:
                print(segment_labels, vid_labels)
            else:
                print("Passed")
        negative_mask = np.zeros(1000, dtype=np.int)
        negative_mask[np.array(
            list(set(range(1000)) - vid_labels_encoded - set(segment_labels)))] = 1

        # Frames. Shape: (frames, 1024)
        tmp = video_features.feature_list['rgb'].feature
        frames = tf.cast(tf.io.decode_raw(
            [x.bytes_list.value[0] for x in tmp], out_type="uint8"), "float32"
        ).numpy()

        # Audio. Shape: (frames, 128)
        tmp = video_features.feature_list['audio'].feature
        audio = tf.cast(tf.io.decode_raw(
            [x.bytes_list.value[0] for x in tmp], out_type="uint8"), "float32"
        ).numpy()

        # Combine: shape(frames, 1152)
        video_features = torch.from_numpy(
            np.concatenate([frames, audio], axis=-1))

        if self.debug:
            print(f"http://data.yt8m.org/2/j/i/{vid[:2]}/{vid}.js")
            print(segment_labels)
            print(segment_start_times)
            print(segment_scores)
            print(video_features.size(0))
            print("=" * 20 + "\n")

        # skip problematic entries
        if segment_start_times.max() > video_features.size(0):
            # print(vid, segment_start_times, features.size(),
            #       segment_scores, segment_labels)
            LOGGER.debug("Skipped one problematic entry.")
            return [], [], [], [], []
        # assert segment_start_times.max() <= features.size(0)

        # Pad agressively
        # if segment_start_times.max() + 5 > features.size(0):
        video_features_padded = torch.cat(
            [
                torch.zeros(
                    self.offset, video_features.size(1),
                    dtype=video_features.dtype
                ),
                video_features,
                torch.zeros(
                    5 + self.offset, video_features.size(1),
                    dtype=video_features.dtype
                )
            ],
            dim=0
        )

        # Create segments
        # shape: (n_segments, 5 + self.offset * 2)
        indices = (
            torch.from_numpy(segment_start_times).unsqueeze(1) +
            torch.arange(5 + 2 * self.offset).unsqueeze(0)
        )
        # shape: (n_segments * (5 + self.offset * 2), 1152) -> (n_segments, 5 + self.offset * 2, 1152)
        segments = torch.index_select(video_features_padded, 0, indices.view(-1)).view(
            indices.size(0), 5 + self.offset * 2, -1
        )

        return (
            # (n_frames, 5 + self.offset * 2, 1152)
            video_features,
            # (n_segments, 5 + self.offset * 2, 1152)
            segments,
            # (n_segments, 1)
            torch.from_numpy(segment_labels).unsqueeze(1),
            # (n_segments, 1)
            torch.from_numpy(segment_scores).unsqueeze(1),
            # (1000,)
            torch.from_numpy(negative_mask)
        )

    def _iterate_through_dataset(self, tf_dataset):
        for row in tf_dataset:
            video_features, segments, segment_labels, segment_scores, negative_mask = (
                self.prepare_one_sample(row)
            )
            for segment, label, score in zip(segments, segment_labels, segment_scores):
                yield video_features, segment, torch.cat([label, score, negative_mask])

    def generator(self, seed):
        if self.epochs == 1:
            # validation
            tf_dataset = tf.data.TFRecordDataset(
                tf.data.Dataset.from_tensor_slices(self.file_paths)
            )
        else:
            tf_dataset = tf.data.TFRecordDataset(
                # tf.data.Dataset.list_files(
                #     "./data/train/*.tfrecord"
                # )
                tf.data.Dataset.from_tensor_slices(
                    self.file_paths
                ).shuffle(
                    100, seed=seed, reshuffle_each_iteration=True
                ).repeat(self.epochs)
            ).shuffle(256, seed=seed, reshuffle_each_iteration=True).repeat(self.epochs)
        for n_example, row in enumerate(self._iterate_through_dataset(tf_dataset)):
            if self.max_examples and self.max_examples == n_example:
                break
            yield row


class YoutubeVideoDataset(YoutubeSegmentDataset):
    def prepare_one_sample(self, row):
        example = tf.train.SequenceExample()
        tmp = example.FromString(row.numpy())
        context, features = tmp.context, tmp.feature_lists

        vid_labels = list(context.feature['labels'].int64_list.value)
        vid_labels_encoded = set([
            self.label_mapping[x] for x in vid_labels if x in self.label_mapping
        ])
        vid = context.feature['id'].bytes_list.value[0].decode('utf8')

        # Skip rows with empty labels for now
        if not vid_labels_encoded:
            # print("Skipped")
            return None, None

        # Expanded Lables: Shape (1000)
        labels = np.zeros(1000, dtype=np.int)
        labels[list(vid_labels_encoded)] = 1

        # Frames. Shape: (frames, 1024)
        tmp = features.feature_list['rgb'].feature
        frames = tf.cast(
            tf.io.decode_raw(
                [x.bytes_list.value[0] for x in tmp],
                out_type="uint8"
            ), "float32"
        ).numpy()

        # Audio. Shape: (frames, 128)
        tmp = features.feature_list['audio'].feature
        audio = tf.cast(tf.io.decode_raw(
            [x.bytes_list.value[0] for x in tmp], out_type="uint8"), "float32"
        ).numpy()

        # Combine: shape(frames, 1152)
        features = torch.from_numpy(np.concatenate([frames, audio], axis=-1))

        if self.debug:
            print(f"http://data.yt8m.org/2/j/i/{vid[:2]}/{vid}.js")
            print(vid_labels_encoded)
            print(features.size(0))
            print("=" * 20 + "\n")

        return (
            features,
            # (1000,)
            torch.from_numpy(labels)
        )

    def _iterate_through_dataset(self, tf_dataset):
        for row in tf_dataset:
            features, labels = (
                self.prepare_one_sample(row)
            )
            if features is None:
                continue
            yield features, labels


class YoutubeTestDataset(YoutubeSegmentDataset):
    def __init__(self, file_paths, seed=939, debug=False,
                 vocab_path="./data/segment_vocabulary.csv",
                 epochs=1, max_examples=None, offset=0,
                 device="cpu", starts_from=6, ends_at=-2):
        super().__init__(
            file_paths=file_paths, seed=seed, debug=debug,
            vocab_path=vocab_path, epochs=epochs,
            max_examples=max_examples, offset=offset
        )
        self.unfold = nn.Unfold(
            kernel_size=(self.offset * 2 + 5, 1),
            padding=(self.offset, 0), stride=(5, 1))
        self.device = device
        self.starts_from = starts_from
        self.ends_at = ends_at

    def prepare_one_sample(self, row):
        example = tf.train.SequenceExample()
        tmp = example.FromString(row.numpy())
        context, features = tmp.context, tmp.feature_lists

        vid = context.feature['id'].bytes_list.value[0].decode('utf8')

        # Frames. Shape: (frames, 1024)
        tmp = features.feature_list['rgb'].feature
        frames = tf.cast(tf.io.decode_raw(
            [x.bytes_list.value[0] for x in tmp], out_type="uint8"), "float32"
        ).numpy()

        # Audio. Shape: (frames, 128)
        tmp = features.feature_list['audio'].feature
        audio = tf.cast(tf.io.decode_raw(
            [x.bytes_list.value[0] for x in tmp], out_type="uint8"), "float32"
        ).numpy()

        # Combine: shape(frames, 1152)
        video_features = torch.from_numpy(np.concatenate(
            [frames, audio], axis=-1))
        # Combine: shape(frames, 1152)
        video_features_padded = torch.from_numpy(np.concatenate(
            [frames, audio], axis=-1))

        # Pad if necessary
        if video_features.size(0) % 5 != 0:
            video_features_padded = torch.cat(
                [
                    video_features_padded,
                    torch.zeros(
                        5 - video_features_padded.size(0) % 5,
                        video_features_padded.size(1),
                        dtype=video_features_padded.dtype
                    )
                ],
                dim=0
            )

        # shape (1, 1152 * (5 + 2 * self.offset), n_segments, 1)
        unfolded = self.unfold(
            video_features_padded.to(self.device).transpose(1, 0)[None, :, :, None])
        # shape (1152, 5 + 2 * self.offset, n_segments)
        unfolded = unfolded.view(
            video_features_padded.size(1), 5 + 2 * self.offset, -1)
        if self.debug:
            assert unfolded.size(-1) == video_features_padded.size(0) // 5
        # shape (n_segments, 1152, 5 + 2 * self.offset, 1152)
        segment_features = unfolded.transpose(0, 2)
        # Truncate tail and head because they are not evaluated
        segment_features = segment_features[self.starts_from:self.ends_at]
        # Combine: shape(n_segments, frames, 1152)
        # video_features = video_features.unsqueeze(
        #     0).repeat(segment_features.size(0), 1, 1)
        return video_features, segment_features, vid

    def _iterate_through_dataset(self, tf_dataset):
        for row in tf_dataset:
            video_features, segment_features, vid = (
                self.prepare_one_sample(row)
            )
            for i, segment_row in enumerate(segment_features):
                yield video_features, segment_row, i+self.starts_from, vid


def collate_videos(batch, pad=0):
    """Batch preparation.

    Pads the sequences
    """
    transposed = list(zip(*batch))
    max_len = max((len(x) for x in transposed[0]))
    data = torch.zeros(
        (len(batch), max_len, transposed[0][0].size(-1)),
        dtype=torch.float
    ) + pad
    masks = torch.zeros((len(batch), max_len), dtype=torch.float)
    for i, row in enumerate(transposed[0]):
        data[i, :len(row)] = row
        masks[i, :len(row)] = 1
    # Labels
    if transposed[1][0] is None:
        return data, masks, None
    labels = torch.stack(transposed[1]).float()
    # print(data.shape, masks.shape, labels.shape)
    return data, masks, labels


def collate_segments(batch, pad=0):
    """Batch preparation.

    Pads the sequences
    """
    #  frames, segment, (label, score, negative_mask)
    transposed = list(zip(*batch))
    max_len = max((len(x) for x in transposed[0]))
    video_data = torch.zeros(
        (len(batch), max_len, transposed[0][0].size(-1)),
        dtype=torch.float
    ) + pad
    video_masks = torch.zeros((len(batch), max_len), dtype=torch.float)
    for i, row in enumerate(transposed[0]):
        video_data[i, :len(row)] = row
        video_masks[i, :len(row)] = 1
    segments = torch.stack(transposed[1]).float()
    labels = torch.stack(transposed[2])
    return video_data, video_masks, segments, labels


def collate_test_segments(batch, pad=0, return_vid=True):
    """Batch preparation for the test dataset
    """
    #  video, segment, vid
    transposed = list(zip(*batch))
    max_len = max((len(x) for x in transposed[0]))
    video_features = torch.zeros(
        (len(batch), max_len, transposed[0][0].size(-1)),
        dtype=torch.float
    ) + pad
    video_masks = torch.zeros((len(batch), max_len), dtype=torch.float)
    for i, row in enumerate(transposed[0]):
        video_features[i, :len(row)] = row
        video_masks[i, :len(row)] = 1
    segment_features = torch.stack(transposed[1], dim=0)
    indices = transposed[2]
    if return_vid:
        vids = transposed[3]
        return video_features, video_masks, segment_features, indices, vids
    return video_features, video_masks, segment_features, indices


def test():
    import glob
    filepaths = list(glob.glob(str("data/train/*.tfrecord")))
    print(filepaths[:2])
    dataset = YoutubeSegmentDataset(filepaths)
    loader = DataLoader(dataset, num_workers=1, batch_size=16)
    max_class = 0
    for i, (segments, label) in enumerate(loader):
        max_class = max(max_class, label[0, :].max())
        if i == 1000:
            print(segments.size(), label.size(), label[0])
            print(max_class)
            break

    filepaths = list(glob.glob(str("data/train/*.tfrecord")))
    print(filepaths[:2])
    dataset = YoutubeVideoDataset(filepaths, epochs=None)
    loader = DataLoader(dataset, num_workers=0,
                        batch_size=16, collate_fn=collate_videos)
    for i, (data, masks, labels) in enumerate(loader):
        if i == 1000:
            print(data.size(), masks.size(), labels.size())
            break


if __name__ == "__main__":
    test()
