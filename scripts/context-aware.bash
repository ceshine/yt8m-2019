SEED=4055 python -m yt8m.train_segment_w_context scripts/segment_with_context.yaml data/cache/video/dbof-3 data/cache/video/nextvlad-2 --fold 3 --name dbof-3_nextvlad-2 --steps 12000
SEED=5055 python -m yt8m.train_segment_w_context scripts/segment_with_context.yaml data/cache/video/dbof-3 data/cache/video/nextvlad-2 --fold 4 --name dbof-3_nextvlad-2

SEED=5455 python -m yt8m.train_segment_w_context scripts/segment_with_context.yaml data/cache/video/nextvlad-2 data/cache/video/nextvlad-2 --fold 5 --name nextvlad-2_x2
SEED=3055 python -m yt8m.train_segment_w_context scripts/segment_with_context.yaml data/cache/video/dbof-3 data/cache/video/dbof-3 --fold 4 --name dbof-3_x2