# Experiment 9-4 : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on

python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('2')" DATASETS.NAMES "('market1501')" MODEL.IF_WITH_CENTER "('yes')" MODEL.METRIC_LOSS_TYPE "('triplet_center')" OUTPUT_DIR "('/home/haoluo/log/gu/reid_baseline_review/Opensource_test/market1501/Experiment-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005')"