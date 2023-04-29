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

With default config requires 6 GB on GPU
```bash
./train.sh -d /Dataset/ -f cfgs/ -a cfgs/salsanext.yml -m salsanet -l ./logs/ -c 0
```

**SalsaNetRec**

With default config requires 7 GB on GPU
```bash
./train.sh -d /Dataset/ -f cfgs/ -a cfgs/salsanext.yml -m salsanet_rec -l ./logs/ -c 0
```
**SalsaNetRecLSTM**

With default config requires 9 GB on GPU
```bash
./train.sh -d /Dataset/ -f cfgs/ -a cfgs/salsanext.yml -m salsanet_rec_lstm -l ./logs/ -c 0
```
**SalsaNext**

With default config requires 10 GB on GPU
```bash
./train.sh -d /Dataset/ -f cfgs/ -a cfgs/salsanext.yml -m salsanext -l ./logs/ -c 0
```
**DDRNet**

Change `MODEL.MOD` inside [`ddrnet23_slim.yml`](./cfgs/ddrnet23_slim.yml) to `'none'`, `'oc'` or `'da'` for different model configurations before run. 

With default config requires 2, 3 & 10 GB on GPU for `none`, `oc` & `da` respectively 
```bash
./train.sh -d /Dataset/ -f cfgs/ -a cfgs/ddrnet23_slim.yml -m ddrnet -l ./logs/ -c 0
```
**SegFormer**

With default config requires 11 GB on GPU
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
./infer.sh -d /Dataset/ -f cfgs/ -l ./logs/ddrnet_aug-set-5+t-z/ -m ddrnet  -p ./logs/infer/ddrnet_aug-set-5+t-z -s valid -c 0
```

In order to calculate metrics for your predicrions, run [`eval.sh`](./eval.sh). Metrics will be saved in `iou.txt` inside your `<predictions folder>`:
```bash
./eval.sh -d /Dataset/ -f cfgs/ -p ./logs/infer/ddrnet_aug-set-5+t-z -s valid
```
## Pretrained Models

### Segmentation models evaluated on DAPS-2 dataset
These are inference results on DAPS-2 for the models listed in the following sections (see our [Paper]())
| Method | $mIoU$, % | $IoU$[human], % | $IoU$[surface], % | $IoU$[obstacle], % | Checkpoint |
|   :---| :---:   |  :---: |    :---:   |    :---:   |    :---:   |
| <td colspan=5>Trained on SemanticKITTI</td>
| SalsaNet | **0.405** | 0.056 | 0.662 | **0.498** | [model](https://drive.google.com/file/d/1TsZxuFdRpss9kV2CXpViqhvlyJZuYjnt/view?usp=share_link) |
| SalsaNetRec | 0.267 | 0.034 | 0.478 | 0.290 | [model](https://drive.google.com/file/d/1JnY0XnhzGXFmVvzVKr_sWg9O4wqFlFvy/view?usp=share_link) |
| SalsaNextRecLSTM | 0.338 | 0.075 | 0.565 | <ins>0.373</ins> | [model](https://drive.google.com/file/d/1UHYc4M704dwjIW02sE7QrS4vZp7RwWx4/view?usp=share_link) |
| SalsaNext | 0.258 | 0.074 | 0.423 | 0.277 | [model](https://drive.google.com/file/d/1cx1YQfC_soAaxMGxO2nZ4_GvMrlb97nV/view?usp=share_link) |
| DDRNet | 0.345 | <ins>0.080</ins> | **0.768** | 0.187 | [model](https://drive.google.com/file/d/1_NcEHq3XHAZHtQkzMfDxaNNZypX1cIh3/view?usp=share_link) |
| DDRNetOC | 0.323 | 0.015 | <ins>0.739</ins> | 0.216 | [model](https://drive.google.com/file/d/1I4g7SOAIEqzQzkdgGVz_eYl5w2-PjLlw/view?usp=share_link) |
| DDRNetDA | <ins>0.369</ins> | 0.040 | 0.705 | 0.361 | [model](https://drive.google.com/file/d/1iWqTvLDKdF_jxtzRNsPP0j1CFfcA4o8Q/view?usp=share_link) |
| Segformer | 0.230 | **0.166** | 0.182 | 0.343 | [model](https://drive.google.com/file/d/1qM-rgoga3KYdGyH9wU2BmRisXjtaJPRH/view?usp=share_link) |
| <td colspan=5>Trained on SemanticKITTI + RELLIS-3D</td> |
| SalsaNet | **0.712** | 0.733 | <ins>0.760</ins> | **0.643** | [model](https://drive.google.com/file/d/1u91Klb_bDpY-tgktIBF_GJoiCdKUxgI-/view?usp=share_link) |
| SalsaNetRec | 0.481 | 0.446 | 0.590 | 0.406 | [model](https://drive.google.com/file/d/1UalGUKak7Ono7d0lN6XPuqL1jsfMn27K/view?usp=share_link) |
| SalsaNextRecLSTM | <ins>0.689</ins> | <ins>0.770</ins> | **0.761** | 0.537 | [model](https://drive.google.com/file/d/1TjrvJt6RHSdc1Mb75x83yji-Mx1HVUAf/view?usp=share_link) |
| SalsaNext | 0.663 | **0.799** | 0.684 | 0.505 | [model](https://drive.google.com/file/d/1OnCw32W0YX4ogf8SedmxutW4tt05i_Ky/view?usp=share_link) |
| DDRNet | 0.580 | 0.697 | 0.535 | 0.509 | [model](https://drive.google.com/file/d/1Y7LU5O54qyMhXNxGL4l_onyVnBXIJEvP/view?usp=share_link) |
| DDRNetOC | 0.616 | 0.724 | 0.586 | <ins>0.539</ins> | [model](https://drive.google.com/file/d/1BFwGFUAzXieXHv-tOaiL430MUb3GrAj6/view?usp=share_link) |
| DDRNetDA | 0.541 | 0.614 | 0.507 | 0.503 | [model](https://drive.google.com/file/d/18-_gxPP-2thy-26JJWwnT9etbro8po7w/view?usp=share_link) |
| Segformer | 0.290 | 0.412 | 0.054 | 0.404 | [model](https://drive.google.com/file/d/1G0sxs1ZKwuKzToV_kCyiawJqPE0fJeJw/view?usp=share_link) |
| <td colspan=5>Trained on DAPS-1</td> |
| SalsaNet | 0.575 | 0.479 | 0.741 | 0.504 | [model](https://drive.google.com/file/d/1N_bElG9oYPvMltCfweorihCr2SSakbQF/view?usp=share_link) |
| SalsaNetRec | 0.609 | <ins>0.596</ins> | <ins>0.748</ins> | 0.482 | [model](https://drive.google.com/file/d/13Tu687Y7XcYmQAkXHDhARSsPfu16NT7t/view?usp=share_link) |
| SalsaNextRecLSTM | <ins>0.624</ins> | 0.532 | **0.783** | 0.557 | [model](https://drive.google.com/file/d/1WKXwYrCGQ5enshJgUU9R_eWB9aL4NDTQ/view?usp=share_link) |
| SalsaNext | **0.643** | **0.706** | 0.620 | **0.602** | [model](https://drive.google.com/file/d/1jxf9nOHtbkoAfj7j3FQg3HDq8Pj8t7lE/view?usp=share_link) |
| DDRNet | 0.516 | 0.354 | 0.737 | 0.458 | [model](https://drive.google.com/file/d/13yxGId_LOZGMB2yuMbzQNlCpwcoWkagw/view?usp=share_link) |
| DDRNetOC | 0.562 | 0.403 | 0.705 | <ins>0.577</ins> | [model](https://drive.google.com/file/d/1nznz3qo71UNmUrnYOS-ErMPptdH0g2qY/view?usp=share_link) |
| DDRNetDA | 0.531 | 0.360 | 0.718 | 0.515 | [model](https://drive.google.com/file/d/1gnoO10xB9KHrQ2YIQB7fLY5pjntvDica/view?usp=share_link) |
| Segformer | 0.393 | 0.355 | 0.418 | 0.407 | [model](https://drive.google.com/file/d/1qhDtSHxbCOZ89RXOTOxV11bSmyxoikqV/view?usp=share_link) |


### Segmentation models trained and evaluated on DAPS-1 dataset
All models are trained with augmentation set 5 with T-Zone (see our [Paper]())
| Method | $mIoU$, % | $IoU$[vehicle], % | $IoU$[human], % | $IoU$[surface], % | $IoU$[obstacle], % | Checkpoint |
|   :---| :---:   |  :---: |    :---:   |    :---:   |    :---:   |    :---:   |
| SalsaNet | <ins>0.867</ins> | **0.880** | 0.646 | <ins>0.989</ins> | <ins>0.953</ins> |  [model](https://drive.google.com/file/d/1N_bElG9oYPvMltCfweorihCr2SSakbQF/view?usp=share_link) |
| SalsaNetRec | 0.850 | 0.836 | <ins>0.680</ins> | 0.974 | 0.908 |  [model](https://drive.google.com/file/d/13Tu687Y7XcYmQAkXHDhARSsPfu16NT7t/view?usp=share_link) |
| SalsaNetRecLSTM | 0.862 | <ins>0.878</ins> | 0.632 | 0.988 | 0.949 |  [model](https://drive.google.com/file/d/1WKXwYrCGQ5enshJgUU9R_eWB9aL4NDTQ/view?usp=share_link) |
| SalsaNext | **0.886** | <ins>0.878</ins> | **0.721** | **0.990** | **0.954** |  [model](https://drive.google.com/file/d/1jxf9nOHtbkoAfj7j3FQg3HDq8Pj8t7lE/view?usp=share_link) |
| DDRNet | 0.690 | 0.773 | 0.126 | 0.977 | 0.886 |  [model](https://drive.google.com/file/d/13yxGId_LOZGMB2yuMbzQNlCpwcoWkagw/view?usp=share_link) |
| DDRNetOC | 0.694 | 0.769 | 0.138 | 0.978 | 0.889 |  [model](https://drive.google.com/file/d/1nznz3qo71UNmUrnYOS-ErMPptdH0g2qY/view?usp=share_link) |
| DDRNetDA | 0.691 | 0.770 | 0.129 | 0.977 | 0.886 |  [model](https://drive.google.com/file/d/1gnoO10xB9KHrQ2YIQB7fLY5pjntvDica/view?usp=share_link) |
| Segformer | 0.530 | 0.495 | 0.052 | 0.909 | 0.665 |  [model](https://drive.google.com/file/d/1qhDtSHxbCOZ89RXOTOxV11bSmyxoikqV/view?usp=share_link) |

### Segmentation models trained and evaluated on SemanticKITTI dataset
All models are trained with augmentation set 5 without T-Zone (see our [Paper]())
| Method | $mIoU$, % | $IoU$[vehicle], % | $IoU$[human], % | $IoU$[surface], % | $IoU$[obstacle], % | Checkpoint |
|   :---| :---:   |  :---: |    :---:   |    :---:   |    :---:   |    :---:   |
| SalsaNet | 0.787 | 0.882 | 0.412 | **0.929** | **0.924** | [model](https://drive.google.com/file/d/1TsZxuFdRpss9kV2CXpViqhvlyJZuYjnt/view?usp=share_link) |
| SalsaNetRec | <ins>0.789</ins> | 0.855 | <ins>0.488</ins> | 0.913 | 0.900 | [model](https://drive.google.com/file/d/1JnY0XnhzGXFmVvzVKr_sWg9O4wqFlFvy/view?usp=share_link) |
| SalsaNetRecLSTM | 0.751 | <ins>0.887</ins> | 0.271 | <ins>0.927</ins> | <ins>0.920</ins> | [model](https://drive.google.com/file/d/1UHYc4M704dwjIW02sE7QrS4vZp7RwWx4/view?usp=share_link) |
| SalsaNext | **0.821** | **0.907** | **0.564** | 0.905 | 0.907 | [model](https://drive.google.com/file/d/1cx1YQfC_soAaxMGxO2nZ4_GvMrlb97nV/view?usp=share_link) |
| DDRNet | 0.692 | 0.750 | 0.225 | 0.901 | 0.893 | [model](https://drive.google.com/file/d/1_NcEHq3XHAZHtQkzMfDxaNNZypX1cIh3/view?usp=share_link) |
| DDRNetOC | 0.687 | 0.739 | 0.222 | 0.900 | 0.889 | [model](https://drive.google.com/file/d/1I4g7SOAIEqzQzkdgGVz_eYl5w2-PjLlw/view?usp=share_link) |
| DDRNetDA | 0.696 | 0.754 | 0.232 | 0.903 | 0.895 | [model](https://drive.google.com/file/d/1iWqTvLDKdF_jxtzRNsPP0j1CFfcA4o8Q/view?usp=share_link) |
| SegFormer | 0.539 | 0.437 | 0.048 | 0.893 | 0.777 | [model](https://drive.google.com/file/d/1qM-rgoga3KYdGyH9wU2BmRisXjtaJPRH/view?usp=share_link) |

### Segmentation models trained and evaluated on SemanticKITTI and RELLIS-3D datasets
All models are trained with augmentation set 5 with T-Zone (see our [Paper]())
| Method | $mIoU$, % | $IoU$[vehicle], % | $IoU$[human], % | $IoU$[surface], % | $IoU$[obstacle], % | Checkpoint |
|   :---| :---:   |  :---: |    :---:   |    :---:   |    :---:   |    :---:   |
| SalsaNet | **0.832** | 0.869 | <ins>0.886</ins> | **0.789** | **0.782** | [model](https://drive.google.com/file/d/1u91Klb_bDpY-tgktIBF_GJoiCdKUxgI-/view?usp=share_link) |
| SalsaNetRec | 0.808 | 0.858 | 0.847 | 0.763 | 0.766 | [model](https://drive.google.com/file/d/1UalGUKak7Ono7d0lN6XPuqL1jsfMn27K/view?usp=share_link) |
| SalsaNetRecLSTM | <ins>0.828</ins> | <ins>0.889</ins> | 0.880 | <ins>0.770</ins> | <ins>0.774</ins> | [model](https://drive.google.com/file/d/1TjrvJt6RHSdc1Mb75x83yji-Mx1HVUAf/view?usp=share_link) |
| SalsaNext | **0.832** | **0.904** | **0.906** | 0.755 | 0.763 | [model](https://drive.google.com/file/d/1OnCw32W0YX4ogf8SedmxutW4tt05i_Ky/view?usp=share_link) |
| DDRNet | 0.706 | 0.759 | 0.654 | 0.685 | 0.725 | [model](https://drive.google.com/file/d/1Y7LU5O54qyMhXNxGL4l_onyVnBXIJEvP/view?usp=share_link) |
| DDRNetOC | 0.705 | 0.749 | 0.652 | 0.693 | 0.728 | [model](https://drive.google.com/file/d/1BFwGFUAzXieXHv-tOaiL430MUb3GrAj6/view?usp=share_link) |
| DDRNetDA | 0.658 | 0.715 | 0.471 | 0.716 | 0.731 | [model](https://drive.google.com/file/d/18-_gxPP-2thy-26JJWwnT9etbro8po7w/view?usp=share_link) |
| SegFormer | 0.533 | 0.448 | 0.423 | 0.616 | 0.643 | [model](https://drive.google.com/file/d/1G0sxs1ZKwuKzToV_kCyiawJqPE0fJeJw/view?usp=share_link) |
