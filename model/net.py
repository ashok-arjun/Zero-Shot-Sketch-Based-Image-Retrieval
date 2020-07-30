'''
This file defines the main model(with attention). This can be used directly at test time to get the model embeddings in the higher dimensional space.

At train time, the ~loss.py~ file contains the decoder components(The GRL, Triplet loss, and the semantic loss), and they will be used to optimize the network.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class AttentionBlock(nn.Module):
  def __init__(self, input_filters = 1024, hidden_filters = 512):

    super(AttentionBlock, self).__init__()
    
    self.attention_net = nn.Sequential(
      nn.Conv2d(input_filters, hidden_filters, kernel_size=1),
      nn.Conv2d(hidden_filters, 1, kernel_size=1)
    )

  def forward(self, feature_maps):
    affine_attention = self.attention_net(feature_maps)
    attention_mask = affine_attention.view(affine_attention.shape[0], -1) # Batch size x (1 x h x w)
    attention_mask = F.softmax(attention_mask, dim = 1) # Softmax is applied over the spatial size of each feature map
    attention_mask = attention_mask.view(affine_attention.shape)
    feature_maps = feature_maps + (feature_maps * attention_mask)
    return feature_maps, attention_mask


class MainModel(nn.Module):
  def __init__(self, pretrained = True, output_embedding_size = 300, use_attention = True):

    super(MainModel, self).__init__()
    shufflenet = torchvision.models.shufflenet_v2_x0_5(pretrained=pretrained, progress=False)
    self.features = nn.Sequential(
        *list(shufflenet.children())[:-1]
    )

    self.use_attention = use_attention
    self.attention_block = AttentionBlock(input_filters = 1024, hidden_filters = 512)
    
    
    self.embed_block = nn.Sequential(
      
      nn.Linear(1024, 1024),
      nn.ReLU(),
      nn.Dropout(0.5),

      nn.Linear(1024, 1024),
      nn.ReLU(),
      nn.Dropout(0.5),

      nn.Linear(1024, output_embedding_size)
    ) #TODO: experiment with the architecture



  def forward(self, x):
    x = self.features(x)

    if self.use_attention:
      x, attention_mask = self.attention_block(x)
    else:
      attention_mask = torch.zeros(x.shape[0], *x.shape[2:])

    x = F.max_pool2d(x, kernel_size=x.size()[2:]) # global pooling to 1024 x 1 x 1, TODO:ablation w/o this layer

    x = x.view(x.shape[0], -1) # Batch size x (1024) 

    x = self.embed_block(x)

    return x, attention_mask


