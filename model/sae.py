import numpy as np
import scipy.linalg
import torch

def SAE(x, s, ld):
  '''
  Inputs:
    x: the sketch matrix of shape 1024 x N
    s: the images matrix of shape 1024 x N
    ld: lambda, the regularization parameter
  Outputs:
    w: the weight matrix mapping the sketches to the images; shape: 1024 x 1024
  '''
  if type(x) == torch.Tensor:
    x = x.cpu().numpy()
  if type(s) == torch.Tensor:
    s = s.cpu().numpy()

  A = np.dot(s, s.transpose())
  B = ld * np.dot(x, x.transpose())
  C = (1+ld) * np.dot(s, x.transpose())
  w = scipy.linalg.solve_sylvester(A,B,C)

  return w