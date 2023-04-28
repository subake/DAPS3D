# Getting Started
This document provides a brief intro of the usage of DAPS3D.
## Training
- Make shure to cteate conda environment and setup wandb.login for training (See [Installation](./INSTALL.md)).

- Select desired augmentation setup inside [`augmentation.yml`](./cfgs/augmentation.yml). You can find more detailed description and different augmentation setups in our [Paper](). 

- We have different configs for our models: [`salsanext.yml`](./cfgs/salsanext.yml), [`ddrnet23_slim.yml`](./cfgs/ddrnet23_slim.yml) & [`segformer.yml`](./cfgs/segformer.yml). You can change hyperparameters there before training.

### Train Commands
This is default command structure for training:
```bash
./train.sh -d <path/to/dataset> \
           -f <path/to/configs> \
           -a <path/to/model_config> \
           -m <model_name> \
           -l <path/to/save/logs> \
           -p <path/to/pretrained/logs> \
           -c <gpu to run>
```
You need to choose `<model_name>` between: `salsanet`,  `salsanet_rec`, `salsanet_rec_lstm`, `salsanext`, `ddrnet` or s`egformer`.

**SalsaNet**
```bash
./train.sh -d /Dataset/ -f cfgs/ -a cfgs/salsanext.yml -m salsanet -l ./logs/ -c 0
```

**SalsaNetRec**
```bash
./train.sh -d /Dataset/ -f cfgs/ -a cfgs/salsanext.yml -m salsanet_rec -l ./logs/ -c 0
```
**SalsaNetRecLSTM**
```bash
./train.sh -d /Dataset/ -f cfgs/ -a cfgs/salsanext.yml -m salsanet_rec_lstm -l ./logs/ -c 0
```
**SalsaNext**
```bash
./train.sh -d /Dataset/ -f cfgs/ -a cfgs/salsanext.yml -m salsanext -l ./logs/ -c 0
```
**DDRNet**

Change `MODEL.MOD` inside [`ddrnet23_slim.yml`](./cfgs/ddrnet23_slim.yml) to `'none'`, `'oc'` or `'da'` for different model configurations before run.  
```bash
./train.sh -d /Dataset/ -f cfgs/ -a cfgs/ddrnet23_slim.yml -m ddrnet -l ./logs/ -c 0
```
**SegFormer**
```bash
./train.sh -d /Dataset/ -f cfgs/ -a cfgs/segformer.yml -m segformer -l ./logs/ -c 0
```

## Inference
This is default command structure for inference:
```bash
./infer.sh -d <path/to/dataset> \
          -f <path/to/configs> \
          -l <path/to/pretrained/logs> \
          -m <model_name> \
          -p <path/to/save/predictions> \
          -s <split to inference> \
          -c <gpu to run>
```
Simple example (for DDRNet model):
```bash
./infer.sh -d /Dataset/ -f cfgs/ -l ./logs/ddrnet_aug5_T/ -m ddrnet  -p ./logs/infer/ddrnet_aug5_T -s valid -c 0
```

In order to calculate metrics for your predicrions, run [`eval.sh`](./eval.sh). Metrics will be saved in `iou.txt` inside your `<predictions folder>`:
```bash
./eval.sh -d /Dataset/ -f cfgs/ -p ./logs/infer/ddrnet_aug5_T -s valid
```
## Pretrained Models
