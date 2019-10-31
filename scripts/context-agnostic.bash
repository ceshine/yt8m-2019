SEED=1213 python -m yt8m.train_pure_segment scripts/pure_segment_dbof.yaml data/cache/video/dbof-3/ --fold 0 --name dbof-3
SEED=1216 python -m yt8m.train_pure_segment scripts/pure_segment_dbof.yaml data/cache/video/dbof-3/ --fold 1 --name dbof-3
SEED=1351 python -m yt8m.train_pure_segment scripts/pure_segment_dbof.yaml data/cache/video/dbof-3/ --fold 2 --name dbof-3

SEED=5696 python -m yt8m.train_pure_segment scripts/pure_segment_nextvlad.yaml data/cache/video/nextvlad-2/ --fold 0 --name nextvlad-2
SEED=1696 python -m yt8m.train_pure_segment scripts/pure_segment_nextvlad.yaml data/cache/video/nextvlad-2/ --fold 1 --name nextvlad-2 --steps 12000
SEED=2396 python -m yt8m.train_pure_segment scripts/pure_segment_nextvlad.yaml data/cache/video/nextvlad-2/ --fold 2 --name nextvlad-2