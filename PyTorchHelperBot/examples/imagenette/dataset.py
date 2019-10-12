from pathlib import Path
from typing import Callable, List, Optional, Dict

import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

from transforms import tensor_transform


N_CLASSES = 1103
DATA_ROOT = Path('./data')


def build_dataframe_from_folder(root: Path, class_map: Optional[Dict] = None):
    if class_map is None:
        new_class_map = {}
    tmp = []
    for subfolder in root.iterdir():
        if class_map is None:
            new_class_map[subfolder.name] = len(new_class_map)
            class_id = new_class_map[subfolder.name]
        else:
            class_id = class_map[subfolder.name]
        for image in subfolder.iterdir():
            tmp.append((image, class_id))
    df = pd.DataFrame(tmp, columns=["image_path", "label"])
    if class_map is None:
        return df, new_class_map
    return df


class TrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_transform: Callable, debug: bool = True):
        super().__init__()
        self._df = df
        self._image_transform = image_transform
        self._debug = debug

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        item = self._df.iloc[idx]
        image = load_transform_image(
            item.image_path, self._image_transform, debug=self._debug)
        target = torch.tensor(item.label).long()
        return image, target


def load_transform_image(
        image_path: Path, image_transform: Callable, debug: bool = False):
    image = cv2.imread(str(image_path.absolute()))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image_transform(image=image)["image"]
    # if debug:
    #     image.save('_debug.jpg')
    tensor = tensor_transform(image=image)["image"]
    return tensor
