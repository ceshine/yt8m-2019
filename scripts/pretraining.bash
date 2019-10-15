SEED=27805 python -m yt8m.train_video nextvlad --steps 200000 --ckpt-interval 10000 --lr 3e-4  --groups 16 --batch-size 48 --n-clusters 64 --max-len 150
mv data/cache/video/baseline_model.pth data/cache/video/nxvlad-2.pth
SEED=17805 python -m yt8m.train_video dbof --steps 100000 --ckpt-interval 10000 --lr 3e-4 --batch-size 128 --max-len 150
mv data/cache/video/baseline_model.pth data/cache/video/dbof-3.pth
SEED=4827 python -m yt8m.train_video dbof --steps 120000 --ckpt-interval 10000 --lr 3e-4 --batch-size 32 --n-mixtures 5
mv data/cache/video/baseline_model.pth data/cache/video/dbof-1.pth
SEED=1635 python -m yt8m.train_video dbof --steps 100000 --ckpt-interval 10000 --lr 4e-4  --batch-size 32 --max-len 200
mv data/cache/video/baseline_model.pth data/cache/video/dbof-2.pth
