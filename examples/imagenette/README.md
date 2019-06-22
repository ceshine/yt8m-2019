# Imagenette Example

## Preparation

Download the [imagenette dataset (full)](https://s3.amazonaws.com/fast-ai-imageclas/imagenette.tgz) and extract into `data` folder. It should now contains two folders: `train` and `val`.

## Training instructions

Run `python main.py -h` to view all the available arguments.

## Some Results

| Size (px) | Epochs | Accuracy | Params | Arch | Log |
|--|--|--|--|--|--|
| 192 | 5 | 86.80% | `--batch-size 64 --lr 5e-3 --mixup-alpha 0` | seresnext50 | [bs64_8680.txt](logs/bs64_8680.txt) |
| 192 | 5 | 86.00% | `--batch-size 64 --lr 5e-3 --mixup-alpha 0.2` | seresnext50| [bs64_mixup02_8600.txt](logs/bs64_mixup02_8600.txt) |
| 192 | 10 | 89.80% | `--batch-size 64 --lr 5e-3 --mixup-alpha 0` | seresnext50| [bs64_e10.txt](logs/bs64_e10.txt) |
