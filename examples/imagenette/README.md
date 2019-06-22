# Imagenette Example

## Preparation

Download the [imagenette dataset (full)](https://s3.amazonaws.com/fast-ai-imageclas/imagenette.tgz) and extract into `data` folder. It should now contains two folders: `train` and `val`.

## Training instructions

Run `python main.py -h` to view all the available arguments.

## Some Local Results

Hardware: i7-7700 + GTX 1070

| Size (px) | Epochs | Accuracy | Params | Arch | Log |
|--|--|--|--|--|--|
| 192 | 5 | 86.80% | `--batch-size 64 --lr 5e-3 --mixup-alpha 0` | seresnext50 | [bs64_8680.txt](logs/bs64_8680.txt) |
| 192 | 5 | 86.00% | `--batch-size 64 --lr 5e-3 --mixup-alpha 0.2` | seresnext50| [bs64_mixup02_8600.txt](logs/bs64_mixup02_8600.txt) |
| 192 | 10 | 89.80% | `--batch-size 64 --lr 5e-3 --mixup-alpha 0` | seresnext50| [bs64_e10.txt](logs/bs64_e10.txt) |

## Google Colab Results

[Notebook Link](https://colab.research.google.com/drive/1NppuVSUvNYIEfL7j3DEOKemhrdZFFPDg)

| Size (px) | Epochs | Accuracy | Params | Arch | Log | Amp | Time |
|--|--|--|--|--|--|--|--|
| 192 | 5 | 85.60% | `--batch-size 64 --lr 5e-3 --mixup-alpha 0` | seresnext50 | [colab_o0_bs64_e5.txt](logs/colab_o1_bs64_e5.txt) |  | 13min 18s |
| 192 | 5 | 84.20% | `--batch-size 64 --lr 5e-3 --mixup-alpha 0 --amp O1` | seresnext50 | [colab_o1_bs64_e5.txt](logs/colab_o1_bs64_e5.txt) | O1 | 9min 59s |