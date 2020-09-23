import time
import datetime
import pytz 

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils 


from model.net import BasicModel
from model.dataloader import Dataloaders
from model.layers import grad_reverse
from evaluate import evaluate
from utils import *

class Trainer():
  def __init__(self, data_dir):
    self.dataloaders = Dataloaders(data_dir)
    self.train_dict = self.dataloaders.train_dict
    self.test_dict = self.dataloaders.test_dict
  
  def train_and_evaluate(self, config, checkpoint=None):

    batch_size = config['batch_size']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataloader = self.dataloaders.get_train_dataloader(batch_size = batch_size, shuffle=True) 
    num_batches = len(train_dataloader) 

    image_model = BasicModel().to(device)
    sketch_model = BasicModel().to(device) 
    
    params = [param for param in image_model.parameters() if param.requires_grad == True]
    params.extend([param for param in sketch_model.parameters() if param.requires_grad == True])   
    optimizer = torch.optim.Adam(params, lr=config['lr'])

    criterion = nn.TripletMarginLoss(margin = 1.0, p = 2)

    if checkpoint:
      load_checkpoint(checkpoint, image_model, sketch_model, optimizer)

    print('Training...')    
    

    for epoch in range(config['epochs']):
      accumulated_triplet_loss = RunningAverage()
      accumulated_iteration_time = RunningAverage()

      epoch_start_time = time.time()

      image_model.train() 
      sketch_model.train()
      
      for iteration, batch in enumerate(train_dataloader):
        time_start = time.time()        

        '''GETTING THE DATA'''
        anchors, positives, negatives, label_embeddings, positive_label_idxs, negative_label_idxs = batch
        anchors = torch.autograd.Variable(anchors.to(device)); positives = torch.autograd.Variable(positives.to(device))
        negatives = torch.autograd.Variable(negatives.to(device)); label_embeddings = torch.autograd.Variable(label_embeddings.to(device))

        '''MAIN NET INFERENCE AND LOSS'''
        pred_sketch_features = sketch_model(anchors)
        pred_positives_features = image_model(positives)
        pred_negatives_features = image_model(negatives)

        triplet_loss = config['triplet_loss_ratio'] * criterion(pred_sketch_features, pred_positives_features, pred_negatives_features)
        accumulated_triplet_loss.update(triplet_loss, anchors.shape[0])        

        '''OPTIMIZATION'''
        optimizer.zero_grad()  
        triplet_loss.backward()
        optimizer.step()  


        '''LOGGER'''
        time_end = time.time()
        accumulated_iteration_time.update(time_end - time_start)

        if iteration % config['print_every'] == 0:
          eta_cur_epoch = str(datetime.timedelta(seconds = int(accumulated_iteration_time() * (num_batches - iteration)))) 
          print(datetime.datetime.now(pytz.timezone('Asia/Kolkata')).replace(microsecond = 0), end = ' ')

          print('Epoch: %d [%d / %d] ; eta: %s' % (epoch, iteration, num_batches, eta_cur_epoch))
          print('Average Triplet loss: %f(%f);' % (triplet_loss, accumulated_triplet_loss()))

        
      '''END OF EPOCH'''
      epoch_end_time = time.time()
      print('Epoch %d complete, time taken: %s' % (epoch, str(datetime.timedelta(seconds = int(epoch_end_time - epoch_start_time)))))
      torch.cuda.empty_cache()

      save_checkpoint({'iteration': iteration + epoch * num_batches, 
                        'image_model': image_model.state_dict(), 
                        'sketch_model': sketch_model.state_dict(),
                        'optim_dict': optimizer.state_dict()},
                         checkpoint_dir = config['checkpoint_dir'])
      print('Saved epoch!')
      print('\n\n\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Training of SBIR')
  parser.add_argument('--data', help='Data directory path. Directory should contain two folders - sketches and photos, along with 2 .txt files for the labels', required = True)
  parser.add_argument('--batch_size', type=int, help='Batch size to process the train sketches/photos', default = 1)
  parser.add_argument('--checkpoint_dir', help='Directory to save checkpoints', required=True)
  # fill later
  args = parser.parse_args()

  trainer = Trainer(args.data_dir)
  trainer.train_and_evaluate(vars(args))
