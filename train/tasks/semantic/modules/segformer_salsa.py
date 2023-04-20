import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from tasks.semantic.modules.segformer_modules.base import BaseModel
from tasks.semantic.modules.segformer_modules.heads import SegFormerHead


class SegFormer(BaseModel):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19) -> None:
        super().__init__(backbone, num_classes)
        self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape

        logits_ = F.softmax(y, dim=1)
        return logits_

def get_seg_model(backbone_cfg, n_classes):
    model = SegFormer(backbone_cfg, n_classes)

    return model


class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def pixel_acc(self, pred, label):
    if pred.shape[2] != label.shape[1] and pred.shape[3] != label.shape[2]:
        pred = F.interpolate(pred, (label.shape[1:]), mode="bilinear")
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

  def forward(self, inputs, labels=None, *args, **kwargs):
    outputs = self.model(inputs, *args, **kwargs)
    loss = self.loss(outputs, labels)
    acc  = 0
    return torch.unsqueeze(loss,0), outputs, acc
    
    
class FullModel_infer(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss):
    super(FullModel_infer, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, inputs, labels=None, *args, **kwargs):
    outputs = self.model(inputs, *args, **kwargs)

    return outputs