# Experiment all tricks with center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss
python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('se_resnext50')" MODEL.PRETRAIN_PATH "('/home/haoluo/.torch/models/se_resnext50_32x4d-a260b3a4.pth')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "('/home/haoluo/data')" OUTPUT_DIR "('/home/haoluo/log/gu/reid_baseline_review/Opensource_test/market1501/Experiment-seresnext50-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005')"