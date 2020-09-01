import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class BasicModel(nn.Module):
  '''Main model for the triplet loss objective'''
  def __init__(self):
    super(BasicModel, self).__init__()
    self.net = torchvision.models.densenet121(pretrained = True, progress = False).features
    self.last_layer = torch.nn.MaxPool2d((7,7)) 
  def forward(self, x):
    return self.last_layer(self.net(x)).view(x.shape[0], -1)