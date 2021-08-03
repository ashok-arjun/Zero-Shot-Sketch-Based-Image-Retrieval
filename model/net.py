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
  
  
def cosine_similarity_loss(x, y):
  cosine_similarity = x.unsqueeze(1).bmm(y.unsqueeze(2)).squeeze()
  cosine_similarity /= x.norm(p=2, dim=1) * y.norm(p=2,dim=1) 
  return torch.mean((1.0-cosine_similarity)/2)


class EmbeddingLossModel(nn.Module):
  '''This model tries to reconstruct the embedding in 300 dimensions(Word2Vec/GloVe) from the output of the Basic Model'''
  def __init__(self):
    super(EmbeddingLossModel, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(1024, 1024),
      nn.ReLU(inplace = True),

      nn.Linear(1024, 1024),
      nn.ReLU(inplace = True),

      nn.Linear(1024, 1024),
      nn.ReLU(inplace = True),

      nn.Linear(1024, 512),
      nn.ReLU(inplace = True),

      nn.Linear(512, 300),
      nn.ReLU(inplace = True)
    )

  def forward(self, x, y):
    return cosine_similarity_loss(self.net(x), y)


class DomainAdversarialNet(nn.Module):
  '''
  This model acts as an adversary to the main model. Tries to classify domains(sketch/image). This enables the main model
  to embed the sketch and image in the same space(i.e. making sure that the distributions of the data(the last layer) generated
  from both the image and sketch model to be similar).
  '''

  def __init__(self):
    super(DomainAdversarialNet, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(1024, 1024),
      nn.BatchNorm1d(1024),
      nn.ReLU(inplace = True),

      nn.Linear(1024, 1024),
      nn.BatchNorm1d(1024),      
      nn.ReLU(inplace = True),

      nn.Linear(1024, 1024),
      nn.BatchNorm1d(1024),
      nn.ReLU(inplace = True),

      nn.Linear(1024, 1024),
      nn.BatchNorm1d(1024),
      nn.ReLU(inplace = True),

      nn.Linear(1024, 1)
    )

  def forward(self, x):
    return torch.sigmoid(self.net(x))
