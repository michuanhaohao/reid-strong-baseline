# Experiment all tricks without center loss without re-ranking: 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on (=raw all trick, softmax_triplet.yml)
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# without center loss
# without re-ranking
python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')"  DATASETS.ROOT_DIR "('/home/haoluo/data')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('/home/haoluo/log/gu/reid_baseline_review/Opensource_test/market1501/Experiment-all-tricks-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on/resnet50_model_120.pth')"