import argparse

import time
import datetime
import pytz 
import os
from PIL import Image

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score
import torch 
import torch.nn as nn

from model.net import BasicModel, DomainAdversarialNet
from model.dataloader import Dataloaders
from utils import *

import pandas as pd

def evaluate(batch_size, dataloaders, images_model, sketches_model, label2index, k = 5, num_display = 2, proj_W = None):
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  images_model = images_model.to(device); sketches_model = sketches_model.to(device)
  images_model.eval(); sketches_model.eval()

  images_dataloader = dataloaders.get_test_dataloader(batch_size = batch_size, section = 'photos', shuffle = False)
  sketches_dataloader = dataloaders.get_test_dataloader(batch_size = batch_size, section = 'sketches', shuffle = False)
  test_dict = {v:k for k,v in dataloaders.test_dict.items()}


  '''IMAGES'''
  print('Processing the images. Batch size: %d; Number of batches: %d' % (batch_size, len(images_dataloader)))

  start_time = time.time()

  image_feature_predictions = []; image_label_indices = []; test_images = []
  image_filenames = []; 
  with torch.no_grad():
    for iteration, batch in enumerate(images_dataloader):
      images, label_indices, filenames = batch 
      images = torch.autograd.Variable(images.to(device))
      pred_features = images_model(images)
      test_images.append(images); image_feature_predictions.append(pred_features); image_label_indices.append(label_indices)  
      image_filenames.extend(list(filenames))
  image_feature_predictions = torch.cat(image_feature_predictions,dim=0)
  image_label_indices = torch.cat(image_label_indices,dim=0)
  test_images = torch.cat(test_images, dim = 0)

  end_time = time.time()

  print('Processed the images. Time taken: %s' % (str(datetime.timedelta(seconds = int(end_time - start_time)))))


  '''SKETCHES'''
  print('Processing the sketches. Batch size: %d; Number of batches: %d' % (batch_size, len(sketches_dataloader)))

  start_time = time.time()

  sketch_feature_predictions = []; sketch_label_indices = []; test_sketches = []
  sketch_filenames = []
  with torch.no_grad():
    for iteration, batch in enumerate(sketches_dataloader):
      sketches, label_indices, filenames = batch 
      sketches = torch.autograd.Variable(sketches.to(device))
      pred_features = sketches_model(sketches)
      sketch_filenames.extend(list(filenames))
      test_sketches.append(sketches); sketch_feature_predictions.append(pred_features); sketch_label_indices.append(label_indices)

  sketch_feature_predictions = torch.cat(sketch_feature_predictions,dim=0)
  sketch_label_indices = torch.cat(sketch_label_indices,dim=0)
  test_sketches = torch.cat(test_sketches, dim = 0)

  end_time = time.time()

  print('Processed the sketches. Time taken: %s' % (str(datetime.timedelta(seconds = int(end_time - start_time)))))


  '''Get labels from label indices'''

  image_labels = [test_dict[idx.cpu().item()] for idx in image_label_indices]
  sketch_labels = [test_dict[idx.cpu().item()] for idx in sketch_label_indices]

  '''Create 2 pandas dataframes'''

  image_df = pd.DataFrame()
  image_df['label'] = image_labels
  image_df['path'] = image_filenames

  sketch_df = pd.DataFrame()
  sketch_df['label'] = sketch_labels
  sketch_df['path'] = sketch_filenames


  '''mAP calculation'''
  image_feature_predictions = image_feature_predictions.cpu().numpy() 
  sketch_feature_predictions = sketch_feature_predictions.cpu().numpy() 
  image_label_indices = image_label_indices.cpu().numpy() 
  sketch_label_indices = sketch_label_indices.cpu().numpy() 

  '''IMPORTANT - SAE''' 
  if proj_W: sketch_feature_predictions = sketch_feature_predictions @ proj_W

  distance = cdist(sketch_feature_predictions, image_feature_predictions, 'minkowski')
  similarity = 1.0/distance 


  similarity_df=pd.DataFrame(data=similarity[0:,0:], index=[i for i in range(similarity.shape[0])], columns=[i for i in range(similarity.shape[1])])

  return image_df, sketch_df, similarity_df