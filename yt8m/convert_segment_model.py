import argparse
import glob
from pathlib import Path

import torch

from .segment_models import SegmentModelWrapper
from .inference import patch


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("model_dir", type=str, default="data/cache/segment/")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    for filepath in glob.glob(str(model_dir / "*.pth")):
        print(filepath)
        filepath = Path(filepath)
        name = filepath.stem
        if name.endswith("_wrapped"):
            continue
        model = torch.load(filepath)
        patch(model)
        new_model = SegmentModelWrapper(model).cpu()
        torch.save(new_model, str(model_dir / (name + "_wrapped.pth")))


if __name__ == "__main__":
    main()
