SEED=4827 python -m yt8m.train_video scripts/video_gated_dbof.yaml
mv $(find data/cache/video/ -name "20*" | head -1) data/cache/video/dbof-3
SEED=1635 python -m yt8m.train_video scripts/video_nextvlad.yaml
mv $(find data/cache/video/ -name "20*" | head -1) data/cache/video/nextvlad-2