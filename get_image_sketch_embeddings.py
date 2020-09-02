import argparse

import time
import datetime
import pytz 
import os
from PIL import Image

import numpy as np
import torch 
import torch.nn as nn

from model.net import BasicModel 
from model.dataloader import Dataloaders
from utils import *

def get_embeddings(data_dir, checkpoint=None, batch_size = 8):
  dataloaders = Dataloaders(data_dir)

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  images_model = BasicModel().to(device)
  sketches_model = BasicModel().to(device) 
  images_model.eval(); sketches_model.eval()
  if checkpoint: load_checkpoint(checkpoint, images_model, sketches_model)

  train_dataloader = dataloaders.get_train_dataloader(batch_size = batch_size, shuffle = False)

  start_time = time.time()
  print('Starting the caching process')
  image_feature_predictions = []; sketch_feature_predictions = [];
  with torch.no_grad():
    for iteration, batch in enumerate(train_dataloader):
      anchors, positives, _, _, _, _ = batch
      anchors = torch.autograd.Variable(anchors.to(device)); positives = torch.autograd.Variable(positives.to(device))
      pred_sketch_features = sketches_model(anchors)
      pred_positives_features = images_model(positives)
      image_feature_predictions.append(pred_positives_features)
      sketch_feature_predictions.append(pred_sketch_features)
  image_feature_predictions = torch.cat(image_feature_predictions,dim=0)
  sketch_feature_predictions = torch.cat(sketch_feature_predictions,dim=0)
  print('Finished the caching process')
  end_time = time.time()

  image_feature_predictions = np.transpose(image_feature_predictions.cpu().numpy(), (1,0)) 
  sketch_feature_predictions = np.transpose(sketch_feature_predictions.cpu().numpy(), (1,0)) 

  return sketch_feature_predictions, image_feature_predictions


