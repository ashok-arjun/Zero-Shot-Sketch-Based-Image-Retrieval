import time
import datetime
import pytz 

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils 
from torch.utils.tensorboard import SummaryWriter
import wandb


from model.net import BasicModel, EmbeddingLossModel
from model.dataloader import Dataloaders
from evaluate import evaluate
from utils import *

class Trainer():
  def __init__(self, data_dir):
    self.dataloaders = Dataloaders(data_dir)
    self.train_dict = self.dataloaders.train_dict
    self.test_dict = self.dataloaders.test_dict
  
  def train_and_evaluate(self, config, checkpoint_file, local):

    batch_size = config['batch_size']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataloader = self.dataloaders.get_train_dataloader(batch_size = batch_size, shuffle=True) 
    num_batches = len(train_dataloader) 

    image_model = BasicModel()
    sketch_model = BasicModel()
    # embedding_model = EmbeddingLossModel()

    image_model = image_model.to(device); sketch_model = sketch_model.to(device); 
    
    # embedding_model = embedding_model.to(device)

    params = [param for param in image_model.parameters() if param.requires_grad == True]
    params.extend([param for param in sketch_model.parameters() if param.requires_grad == True])   
    # params.extend([param for param in embedding_model.parameters() if param.requires_grad == True])   

    print('A total of %d parameters are present in the models' % (len(params)))

    optimizer = torch.optim.Adam(params, lr=config['lr'])
    criterion = nn.TripletMarginLoss(margin = 1.0, p = 2)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = config['lr_scheduler_step_size'], gamma = 0.1)
    for i in range(config['start_epoch']):
      lr_scheduler.step() 
    wandb_step = config['start_epoch'] * num_batches -1 
    print('Training...')    

    accumulated_triplet_loss = RunningAverage()
    # accumulated_embedding_loss = RunningAverage()

    for epoch in range(config['start_epoch'], config['epochs']):

      accumulated_iteration_time = RunningAverage()

      epoch_start_time = time.time()

      image_model = image_model.train(); 
      sketch_model.train(); 
      # embedding_model.train()
      
      for iteration, batch in enumerate(train_dataloader):
        wandb_step += 1

        time_start = time.time()        

        optimizer.zero_grad()

        '''GETTING THE DATA'''
        anchors, positives, negatives, label_embeddings, positive_label_idxs, negative_label_idxs = batch
        anchors = torch.autograd.Variable(anchors.to(device)); positives = torch.autograd.Variable(positives.to(device))
        negatives = torch.autograd.Variable(negatives.to(device)); label_embeddings = torch.autograd.Variable(label_embeddings.to(device))

        '''INFERENCE AND LOSS'''
        pred_sketch_features = sketch_model(anchors)
        pred_positives_features = image_model(positives)
        pred_negatives_features = image_model(negatives)

        # pred_positives_embedding_loss = embedding_model(pred_positives_features, label_embeddings)
        # pred_sketch_embedding_loss = embedding_model(pred_sketch_features, label_embeddings)
        # pred_negatives_embedding_loss = embedding_model(pred_negatives_features, label_embeddings)

        triplet_loss = criterion(pred_sketch_features, pred_positives_features, pred_negatives_features)
        # embedding_loss = pred_positives_embedding_loss + pred_sketch_embedding_loss - pred_negatives_embedding_loss

        accumulated_triplet_loss.update(triplet_loss, batch_size)
        # accumulated_embedding_loss.update(embedding_loss, batch_size)
        
        '''OPTIMIZATION'''
        total_loss = triplet_loss
        # total_loss += embedding_loss
        total_loss.backward()
        optimizer.step()

        '''TIME UTILS & PRINTING'''
        time_end = time.time()
        accumulated_iteration_time.update(time_end - time_start)
        eta_cur_epoch = str(datetime.timedelta(seconds = int(accumulated_iteration_time() * (num_batches - iteration))))

        if iteration % config['print_every'] == 0:
          print(datetime.datetime.now(pytz.timezone('Asia/Kolkata')).replace(microsecond = 0), end = ' ')
          print('Epoch: %d [%d / %d] ; eta: %s' % (epoch, iteration, num_batches, eta_cur_epoch))
          print('Triplet loss: %f(%f);' % (triplet_loss, accumulated_triplet_loss()))
          # print('Embedding loss: %f(%f)' % (embedding_loss, accumulated_embedding_loss()))

          wandb.log({'Average Triplet loss': accumulated_triplet_loss()}, step = wandb_step)
          # wandb.log({'Average Embedding loss': accumulated_embedding_loss()}, step = wandb_step)
          
      '''END OF EPOCH'''
      epoch_end_time = time.time()
      print('Epoch %d complete, time taken: %s' % (epoch, str(datetime.timedelta(seconds = int(epoch_end_time - epoch_start_time)))))
      lr_scheduler.step()
      torch.cuda.empty_cache()

      sketches, image_grids, test_mAP = evaluate(config, self.dataloaders.get_full_train_dataloader, image_model, sketch_model, self.dataloaders.train_dict)

      wandb.log({'Sketches': [wandb.Image(image) for image in sketches]}, step = wandb_step)
      wandb.log({'Retrieved Images': [wandb.Image(image) for image in image_grids]}, step = wandb_step)

      wandb.log({'Average Training mAP': test_mAP}, step = wandb_step)
      save_checkpoint({'iteration': wandb_step, 
                        'image_model': image_model.state_dict(), 
                        'sketch_model': sketch_model.state_dict(),
                        'optim_dict': optimizer.state_dict()},
                        checkpoint_dir = 'experiments/', save_to_cloud = True)
      print('Saved epoch to cloud!')
      print('\n\n\n')