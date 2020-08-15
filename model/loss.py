'''
This file defines the decoder components(GRL, Triplet loss, semantic embedding loss), and is used to optimize the network while training
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class GradReverse(torch.autograd.Function):
    '''GRL layer from https://arxiv.org/abs/1409.7495'''

    @staticmethod
    def forward(ctx, x, lambd=0.5):
        # ctx is a context object that can be used to stash information
        # for backward computation,like intermediate parameters
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # We return the negated input multiplied by lambd
        # None is the backward for the lambd argument
        return ctx.lambd * grad_output.neg(), None

def grad_reverse(x, lambd=0.5):
    return GradReverse.apply(x, lambd)


def cosine_loss(x, y):
  cosine_similarity = x.unsqueeze(1).bmm(y.unsqueeze(2)).squeeze() # batch matrix multiply the embeddings; gives (Batch size x 1)
  norm_x = x.norm(p=2, dim=1) # Second order norm
  norm_y = y.norm(p=2, dim=1) # Second order norm
  cosine_similarity = cosine_similarity / (norm_x * norm_y)
  loss = (1 - cosine_similarity) / 2
  return loss


class SemanticLoss(nn.Module):
  '''
  Semantic loss between the reconstructed embedding and the original word2vec/other embedding
  '''  

  def __init__(self, input_size = 300, embedding_size = 300):
    super(SemanticLoss, self).__init__()

    self.net = nn.Sequential(
      nn.Linear(input_size, 1024),
      nn.ReLU(True),
      nn.Dropout(0.5),

      nn.Linear(1024, embedding_size)
    )

  def forward(self, input, target):
    input = self.net(input)
    return cosine_loss(input, target)


class DomainLoss(nn.Module):
  '''
  Domain loss for the domain prediction(this acts as a domain adversary to our main model by means of the GRL) 
  '''

  def __init__(self, input_size = 300, hidden_size = 1024):
    super(DomainLoss, self).__init__()

    self.net = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.ReLU(True),
      nn.Dropout(0.5),

      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),

      nn.Linear(hidden_size, 1)
    )

  def forward(self, input, target):
    input = torch.sigmoid(self.net(input)).squeeze()
    return F.binary_cross_entropy(input, target)


class DetangledJointDomainLoss(nn.Module):
  def __init__(self, input_size = 300, grl_lambda = 0.5, w_dom = 0.25, w_triplet = 0.25, w_sem = 0.25, device = torch.device('cpu')):
    super(DetangledJointDomainLoss, self).__init__()

    self.grl_lambda = grl_lambda
    
    self.w_dom = w_dom
    self.w_triplet = w_triplet
    self.w_sem = w_sem

    self.domain_loss = DomainLoss(input_size = input_size)
    self.semantic_loss = SemanticLoss(input_size = input_size, embedding_size = 300) # 300 represents the word2vec size
    self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2) # Triplet loss with margin 1.0 and distance of second order

    self.device = device

  def forward(self, anchor_output, positive_output, negative_output, embedding, epoch):
    '''
    Returns the loss, by combining the three losses, epoch parameter is to anneal GRL lambda over time
    '''

#     loss_semantic = self.semantic_loss(anchor_output, embedding)
#     loss_semantic += self.semantic_loss(positive_output, embedding)  
#     loss_semantic += self.semantic_loss(grad_reverse(negative_output, self.grl_lambda), embedding)
#     loss_semantic = loss_semantic.mean()

    loss_semantic = torch.tensor(0.0).to(self.device)

    loss_triplet = self.triplet_loss(anchor_output, positive_output, negative_output)

    # Create targets for the domain loss(INDIRECTLY adversarial for the main model - as imposed by the GRL after every output)
    batch_size = anchor_output.shape[0]
    targets_sketch = torch.zeros(batch_size).to(self.device)
    targets_photos = torch.ones(batch_size).to(self.device)

#     if epoch < 5:
#       lmbda = 0
#     elif epoch < 25:
#       lmbda = (epoch-5)/20.0
#     else:
#       lmbda = 1.0

    lmbda = epoch/25.0 if epoch <= 25 else 1.0


    loss_domain = self.domain_loss(grad_reverse(anchor_output, lmbda), targets_sketch) + self.domain_loss(grad_reverse(positive_output, lmbda), targets_photos) + self.domain_loss(grad_reverse(negative_output, lmbda), targets_photos)
    loss_domain /= 3.0


    total_loss = self.w_dom * loss_domain + self.w_sem * loss_semantic + self.w_triplet * loss_triplet # Our network minimizes this loss

    return total_loss, loss_domain, loss_triplet, loss_semantic
