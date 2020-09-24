import numpy as np
import os
import torch
from torchvision.utils import make_grid

class RunningAverage():
  def __init__(self):
    self.count = 0
    self.sum = 0

  def update(self, value, n_items = 1):
    self.sum += value * n_items
    self.count += n_items

  def __call__(self):
    return self.sum/self.count  


def save_checkpoint(state, checkpoint_dir):
    file_name = 'last.pth.tar'
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    torch.save(state, os.path.join(checkpoint_dir, file_name))

def load_checkpoint(checkpoint, image_model, sketch_model, domain_model=None, optimizer=None):
    if not os.path.exists(checkpoint):
        raise Exception("File {} doesn't exist".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    print('Loading the models from the end of net iteration %d' % (checkpoint['iteration']))
    image_model.load_state_dict(checkpoint['image_model'])
    sketch_model.load_state_dict(checkpoint['sketch_model'])
    if domain_model: domain_model.load_state_dict(checkpoint['domain_model'])
    if optimizer:
      optimizer.load_state_dict(checkpoint['optim_dict'])

def get_sketch_images_grids(sketches, images, similarity_scores, k, num_display):

  if num_display == 0 or k == 0:
    return None, None
  num_sketches = sketches.shape[0]
  indices = np.random.choice(num_sketches, num_display)

  cur_sketches = sketches[indices]; cur_similarities = similarity_scores[indices]
  top_k_similarity_indices  = np.flip(np.argsort(cur_similarities, axis = 1)[:, -k:], axis = 1).copy()
  top_k_similarity_values = np.flip(np.sort(cur_similarities, axis = 1)[:,-k:], axis = 1).copy()
  matched_images = [images[top_k_similarity_indices[i]] for i in range(num_display)]

  list_of_sketches = [np.transpose(cur_sketches[i].cpu().numpy(), (1,2,0)) for i in range(num_display)]
  list_of_image_grids = [np.transpose(make_grid(matched_images[i], nrow = k).cpu().numpy(), (1,2,0)) for i in range(num_display)]

  return list_of_sketches, list_of_image_grids
