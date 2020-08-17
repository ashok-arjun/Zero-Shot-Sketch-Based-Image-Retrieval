import time
import datetime
import pytz 

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score
import torch 
import torch.nn as nn
import wandb

from model.net import MainModel
from utils import *

def evaluate(config, dataloader_fn, images_model, sketches_model, label2index, k = 5, num_display = 2):
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  images_model = images_model.to(device); sketches_model = sketches_model.to(device)
  images_model.eval(); sketches_model.eval()

  batch_size = config['test_batch_size']
  images_dataloader = dataloader_fn(batch_size = batch_size, section = 'photos', shuffle = False)
  sketches_dataloader = dataloader_fn(batch_size = batch_size, section = 'sketches', shuffle = False)

  '''IMAGES'''
  # print('Processing the images. Batch size: %d; Number of batches: %d' % (batch_size, len(images_dataloader)))

  start_time = time.time()

  image_feature_predictions = []; image_label_indices = []; test_images = []
  with torch.no_grad():
    for iteration, batch in enumerate(images_dataloader):
      images, label_indices = batch 
      images = torch.autograd.Variable(images.to(device))
      pred_features,_ = images_model(images)
      test_images.append(images); image_feature_predictions.append(pred_features); image_label_indices.append(label_indices)  
  image_feature_predictions = torch.cat(image_feature_predictions,dim=0)
  image_label_indices = torch.cat(image_label_indices,dim=0)
  test_images = torch.cat(test_images, dim = 0)

  end_time = time.time()

  # print('Processed the images. Time taken: %s' % (str(datetime.timedelta(seconds = int(end_time - start_time)))))


  '''SKETCHES'''
  # print('Processing the sketches. Batch size: %d; Number of batches: %d' % (batch_size, len(sketches_dataloader)))

  start_time = time.time()

  sketch_feature_predictions = []; sketch_label_indices = []; test_sketches = []
  with torch.no_grad():
    for iteration, batch in enumerate(sketches_dataloader):
      sketches, label_indices = batch 
      sketches = torch.autograd.Variable(sketches.to(device))
      pred_features,_ = sketches_model(sketches)
      test_sketches.append(sketches); sketch_feature_predictions.append(pred_features); sketch_label_indices.append(label_indices)

  sketch_feature_predictions = torch.cat(sketch_feature_predictions,dim=0)
  sketch_label_indices = torch.cat(sketch_label_indices,dim=0)
  test_sketches = torch.cat(test_sketches, dim = 0)

  end_time = time.time()

  # print('Processed the sketches. Time taken: %s' % (str(datetime.timedelta(seconds = int(end_time - start_time)))))

  '''mAP calculation'''
  image_feature_predictions = image_feature_predictions.cpu().numpy() 
  sketch_feature_predictions = sketch_feature_predictions.cpu().numpy() 
  image_label_indices = image_label_indices.cpu().numpy() 
  sketch_label_indices = sketch_label_indices.cpu().numpy() 

  distance = cdist(sketch_feature_predictions, image_feature_predictions, 'minkowski')
  similarity = 1.0/distance 

  is_correct_label_index = 1 * (np.expand_dims(sketch_label_indices, axis = 1) == np.expand_dims(image_label_indices, axis = 0))

  average_precision_scores = []
  for i in range(sketch_label_indices.shape[0]):
    average_precision_scores.append(average_precision_score(is_correct_label_index[i], similarity[i])) 
  average_precision_scores = np.array(average_precision_scores)

  index2label = {v: k for k, v in label2index.items()}
  for cls in set(sketch_label_indices):
    print('Class: %s, mAP: %f' % (index2label[cls], average_precision_scores[sketch_label_indices == cls].mean()))

  mean_average_precision = average_precision_scores.mean()

  sketches, image_grids = get_sketch_images_grids(test_sketches, test_images, similarity, k, num_display)

  return sketches, image_grids, mean_average_precision