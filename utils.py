import numpy as np
import os
import torch
import wandb
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


def save_checkpoint(state, checkpoint_dir, save_to_cloud = False):
    prefix = 'state'
    torch.save(state, os.path.join(checkpoint_dir, prefix + '_last.pth.tar'))    
    if save_to_cloud:
      torch.save(state, os.path.join(wandb.run.dir, prefix + "_last.pth.tar"))
      wandb.save(prefix + '_last.pth.tar')

def load_checkpoint(checkpoint, image_model, sketch_model, loss_model, optimizer=None):
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    image_model.load_state_dict(checkpoint['image_model'])
    sketch_model.load_state_dict(checkpoint['sketch_model'])
    loss_model.load_state_dict(checkpoint['loss_model'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def get_sketch_images_grids(sketches, images, similarity_scores, k, num_display):

  num_sketches = sketches.shape[0]
  indices = np.random.choice(num_sketches, num_display)

  cur_sketches = sketches[indices]; cur_similarities = similarity_scores[indices]
  top_k_similarity_indices = -np.argsort(-cur_similarities, axis = 1)[:, :k]
  matched_images = [images[top_k_similarity_indices[i]] for i in range(num_display)]

  list_of_sketches = [np.transpose(sketches[i].cpu().numpy(), (1,2,0)) for i in range(num_display)]
  list_of_image_grids = [np.transpose(make_grid(matched_images[i], nrow = k).cpu().numpy(), (1,2,0)) for i in range(num_display)]

  return list_of_sketches, list_of_image_grids