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

def stack_into_tensor(list_x):
  '''Takes a list of batched tensors, stacks them and reshapes them to (num_items, 1)'''
  x = torch.stack(list_x)
  x = x.view(x.shape[0] * x.shape[1], -1)
  return x  

def evaluate(images_dataloader, sketches_dataloader, images_model, sketches_model, label2index): 
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  images_model = images_model.to(device); sketches_model = sketches_model.to(device)
  images_model.eval(); sketches_model.eval()


  '''IMAGES'''
  print('Processing the images. Batch size: %d; Number of batches: %d' % (batch_size, len(images_dataloader)))

  start_time = time.time()

  image_feature_predictions = []; image_label_indices = []

  with torch.no_grad():
    for iteration, batch in enumerate(images_dataloader):
      images, label_indices = batch 
      images = torch.autograd.Variable(images.to(device))
      pred_features = images_model(images)
      image_feature_predictions.append(pred_features); image_label_indices.append(label_indices)  

  image_feature_predictions = stack_into_tensor(image_feature_predictions)
  image_label_indices = stack_into_tensor(image_label_indices)

  end_time = time.time()

  print('Processed the images. Time taken: %s' % (str(datetime.timedelta(seconds = int(end_time - start_time)))))


  '''SKETCHES'''
  print('Processing the sketches. Batch size: %d; Number of batches: %d' % (batch_size, len(sketches_dataloader)))

  start_time = time.time()

  sketch_feature_predictions = []; sketch_label_indices = []
  with torch.no_grad():
    for iteration, batch in enumerate(sketches_dataloader):
      sketches, label_indices = batch 
      sketches = torch.autograd.Variable(sketches.to(device))
      pred_features = sketches_model(sketches)
      sketch_feature_predictions.append(pred_features); sketch_label_indices.append(label_indices)

  sketch_feature_predictions = stack_into_tensor(sketch_feature_predictions)
  sketch_label_indices = stack_into_tensor(sketch_label_indices)

  end_time = time.time()

  print('Processed the sketches. Time taken: %s' % (str(datetime.timedelta(seconds = int(end_time - start_time)))))

  '''mAP calculation'''
  image_feature_predictions = image_feature_predictions.cpu().numpy() 
  sketch_feature_predictions = sketch_feature_predictions.cpu().numpy() 
  image_label_indices = image_label_indices.cpu().numpy() 
  sketch_label_indices = sketch_label_indices.cpu().numpy() 

  distance = cdist(sketch_feature_predictions, image_feature_predictions, 'euclidean')
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

  print('mAP: %f' % (mean_average_precision))
  return mean_average_precision