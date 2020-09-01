import numpy as np
import scipy
import torch

def SAE(x, s, ld):

  if type(x) == torch.Tensor:
    x = x.cpu().numpy()
  if type(s) == torch.Tensor:
    s = s.cpu().numpy()

  A = np.dot(s, s.transpose())
	B = ld * np.dot(x, x.transpose())
	C = (1+ld) * np.dot(s, x.transpose())
	w = scipy.linalg.solve_sylvester(A,B,C)

  if type(x) == torch.Tensor:
    w = torch.from_numpy(w)
	return w