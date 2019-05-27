# Bags of Tricks and A Strong ReID Baseline
Paper: "Bag of Tricks and A Strong Baseline for Deep Person Re-identification"[[pdf]](https://arxiv.org/abs/1903.07071)

The codes are expanded on a [ReID-baseline](https://github.com/L1aoXingyu/reid_baseline) , which is open sourced by our co-first author [Xingyu Liao](https://github.com/L1aoXingyu).

Another re-implement is developed by python2.7 and pytorch0.4. [link](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks)

```
@inproceedings{luo2019bag,
  title={Bag of Tricks and A Strong Baseline for Deep Person Re-identification},
  author={Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2019}
}
```

## Authors
- [Hao Luo](https://github.com/michuanhaohao)
- [Youzhi Gu](https://github.com/shaoniangu)
- [Xingyu Liao](https://github.com/L1aoXingyu)
- [Shenqi Lai](https://github.com/xiaolai-sqlai)

We support
- [x] easy dataset preparation
- [x] end-to-end training and evaluation
- [x] high modular management

Bag of tricks
- Warm up learning rate
- Random erasing augmentation
- Label smoothing
- Last stride
- BNNeck
- Center loss

## TODO list
In the future, we will
- [] support more datasets
- [] support more models
- [] speed up inference
- [] support multi-gpus training
- [] explore more tricks


## Pipeline
<div align=center>
<img src='imgs/pipeline.jpg' width='800'>
</div>

## Results (rank1/mAP)
| Model | Market1501 | DukeMTMC-reID |
| --- | -- | -- |
| Standard baseline | 87.7 (74.0) |  79.7 (63.8) |
| +Warmup | 88.7 (75.2) |  80.6(65.1) |
| +Random erasing augmentation | 91.3 (79.3) |  81.5 (68.3) |
| +Label smoothing | 91.4 (80.3) |  82.4 (69.3) |
| +Last stride=1 | 92.0 (81.7) | 82.6 (70.6) |
| +BNNeck | 94.1 (85.7) | 86.2 (75.9) |
| +Center loss | 94.5 (85.9) | 86.4 (76.4) |
| +Reranking | 95.4 (94.2) | 90.3 (89.1) |

| Backbone | Market1501 | DukeMTMC-reID |
| --- | -- | -- |
| ResNet18 | 91.7 (77.8) |  82.5 (68.8) |
| ResNet34 | 92.7 (82.7) |  86.4(73.6) |
| ResNet50 | 94.5 (85.9) | 86.4 (76.4) |
| ResNet101 | 94.5 (87.1) |  87.6 (77.6) |
| ResNet152 | 80.9 (59.0) | 87.5 (78.0) |
| SeResNet50 | 94.4 (86.3) | 86.4 (76.5) |
| SeResNet101 | 94.6 (87.3) | 87.5 (78.0) |
| SeResNeXt50 | 94.9 (87.6) | 88.0 (78.3) |
| SeResNeXt101 | 95.0 (88.0) | 88.4 (79.0) |

[model(Market1501)](https://drive.google.com/open?id=1hn0sXLZ5yJcxtmuY-ItQfYD7hBtHwt7A)

[model(DukeMTMC-reID)](https://drive.google.com/open?id=1LARvQe-gUbflbanidUM0keKmHoKTpLUj)

## Get Started
The designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder's purpose by yourself.

1. `cd` to folder where you want to download this repo

2. Run `git clone https://github.com/michuanhaohao/reid-strong-baseline.git`

3. Install dependencies:
    - [pytorch>=0.4](https://pytorch.org/)
    - torchvision
    - [ignite=0.1.2](https://github.com/pytorch/ignite) (Note: V0.2.0 may result in an error)
    - [yacs](https://github.com/rbgirshick/yacs)

4. Prepare dataset

    Create a directory to store reid datasets under this repo or outside this repo. Remember to set your path to the root of the dataset in `config/defaults.py` for all training and testing or set in every single config file in `configs/` or set in every single command.

    You can create a directory to store reid datasets under this repo via

    ```bash
    cd reid-strong-baseline
    mkdir data
    ```

    （1）Market1501

    * Download dataset to `data/` from http://www.liangzheng.org/Project/project_reid.html
    * Extract dataset and rename to `market1501`. The data structure would like:

    ```bash
    data
        market1501 # this folder contains 6 files.
            bounding_box_test/
            bounding_box_train/
            ......
    ```
    （2）DukeMTMC-reID

    * Download dataset to `data/` from https://github.com/layumi/DukeMTMC-reID_evaluation#download-dataset
    * Extract dataset and rename to `dukemtmc-reid`. The data structure would like:

    ```bash
    data
        dukemtmc-reid
        	DukeMTMC-reID # this folder contains 8 files.
            	bounding_box_test/
            	bounding_box_train/
            	......
    ```

5. Prepare pretrained model if you don't have

    （1）Resnet

    ```python
    from torchvision import models
    models.resnet50(pretrained=True)
    ```
    （2）Senet

    ```python
    import torch.utils.model_zoo as model_zoo
    model_zoo.load_url('the pth you want to download (specific urls are listed in  ./modeling/backbones/senet.py)')
    ```
    Then it will automatically download model in `~/.torch/models/`, you should set this path in `config/defaults.py` for all training or set in every single training config file in `configs/` or set in every single command.

    （3）Load your self-trained model

    If you want to continue your train process based on your self-trained model, you can change the configuration `PRETRAIN_CHOICE` from 'imagenet' to 'self' and set the `PRETRAIN_PATH` to your self-trained model. We offer `Experiment-pretrain_choice-all_tricks-tri_center-market.sh` as an example. 

6. If you want to know the detailed configurations and their meaning, please refer to `config/defaults.py`. If you want to set your own parameters, you can follow our method: create a new yml file, then set your own parameters.  Add `--config_file='configs/your yml file'` int the commands described below, then our code will merge your configuration. automatically.

## Train
You can run these commands in  `.sh ` files for training different datasets of differernt loss.  You can also directly run code `sh *.sh` to run our demo after your custom modification.

1. Market1501, cross entropy loss + triplet loss

```bash
python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('market1501')" OUTPUT_DIR "('your path to save checkpoints and logs')"
```

2. DukeMTMC-reID, cross entropy loss + triplet loss + center loss


```bash
python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('dukemtmc')" OUTPUT_DIR "('your path to save checkpoints and logs')"
```

## Test
You can test your model's performance directly by running these commands in `.sh ` files after your custom modification. You can also change the configuration to determine which feature of BNNeck is used and whether the feature is normalized (equivalent to use Cosine distance or Euclidean distance) for testing.

Please replace the data path of the model and set the `PRETRAIN_CHOICE` as 'self' to avoid time consuming on loading ImageNet pretrained model.

1. Test with Euclidean distance using feature before BN without re-ranking,.

```bash
python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('market1501')" TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('no')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('your path to trained checkpoints')"
```
2. Test with Cosine distance using feature after BN without re-ranking,.

```bash
python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('market1501')" TEST.NECK_FEAT "('after')" TEST.FEAT_NORM "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('your path to trained checkpoints')"
```
3. Test with Cosine distance using feature after BN with re-ranking

```bash
python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('dukemtmc')" TEST.NECK_FEAT "('after')" TEST.FEAT_NORM "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.RE_RANKING "('yes')" TEST.WEIGHT "('your path to trained checkpoints')"
```

