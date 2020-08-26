import time
import datetime
import pytz 

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils 
import wandb


from model.net import BasicModel, DomainAdversarialNet
from model.dataloader import Dataloaders
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

    domain_net = DomainAdversarialNet().to(device).train()    
    
    params = [param for param in image_model.parameters() if param.requires_grad == True]
    params.extend([param for param in sketch_model.parameters() if param.requires_grad == True])   
    optimizer = torch.optim.Adam(params, lr=config['lr'])

    domain_optim = torch.optim.Adam(domain_net.parameters(), lr = config['lr'] * 1e1)

    criterion = nn.TripletMarginLoss(margin = 1.0, p = 2)
    domain_criterion = nn.BCELoss()

    wandb_step = config['start_epoch'] * num_batches -1 

    if checkpoint:
      load_checkpoint(checkpoint, image_model, sketch_model, domain_net, optimizer, domain_optim)

    print('Training...')    
    

    for epoch in range(config['start_epoch'], config['epochs']):
      accumulated_triplet_loss = RunningAverage()
      accumulated_image_domain_loss = RunningAverage()
      accumulated_sketch_domain_loss = RunningAverage()
      accumulated_iteration_time = RunningAverage()

      epoch_start_time = time.time()

      image_model.train() 
      sketch_model.train() 
      
      for iteration, batch in enumerate(train_dataloader):
        wandb_step += 1
        time_start = time.time()        

        '''GETTING THE DATA'''
        anchors, positives, negatives, label_embeddings, positive_label_idxs, negative_label_idxs = batch
        anchors = torch.autograd.Variable(anchors.to(device)); positives = torch.autograd.Variable(positives.to(device))
        negatives = torch.autograd.Variable(negatives.to(device)); label_embeddings = torch.autograd.Variable(label_embeddings.to(device))

        '''INFERENCE AND LOSS'''
        pred_sketch_features = sketch_model(anchors)
        pred_positives_features = image_model(positives)
        pred_negatives_features = image_model(negatives)

        triplet_loss = criterion(pred_sketch_features, pred_positives_features, pred_negatives_features)
        accumulated_triplet_loss.update(triplet_loss, anchors.shape[0])        

        '''DOMAIN ADVERSARIAL TRAINING''' # vannila GANs for now. Later - add randomness in outputs of generator, or lower the label

        '''DEFINE TARGETS'''
          
        image_domain_targets = torch.full((anchors.shape[0],1), 1, dtype=torch.float, device=device)
        sketch_domain_targets = torch.full((anchors.shape[0],1), 0, dtype=torch.float, device=device)
          
        '''ALLIED + OPTIMIZATION'''
        allied_loss_sketches = config['domain_loss_ratio'] * domain_criterion(domain_net(pred_sketch_features), image_domain_targets)
        if epoch < 5:
          allied_loss_sketches = 0
        elif epoch < 25:
          allied_loss_sketches *= epoch/25          
          
        optimizer.zero_grad()  
        total_loss = triplet_loss + allied_loss_sketches
        total_loss.backward()
        optimizer.step()  
        
        '''ADVERSARIAL + OPTIMIZATION'''
        domain_pred_p_images = domain_net(pred_positives_features.detach())
        domain_pred_n_images = domain_net(pred_negatives_features.detach())
        domain_pred_sketches = domain_net(pred_sketch_features.detach())


        domain_loss_images = domain_criterion(domain_pred_p_images, image_domain_targets) + domain_criterion(domain_pred_n_images, image_domain_targets)
        accumulated_image_domain_loss.update(domain_loss_images, anchors.shape[0])
        domain_loss_sketches = domain_criterion(domain_pred_sketches, sketch_domain_targets)
        accumulated_sketch_domain_loss.update(domain_loss_sketches, anchors.shape[0])
        
        domain_optim.zero_grad()         
        total_domain_loss = domain_loss_images + domain_loss_sketches        
        total_domain_loss.backward()        
        domain_optim.step()                          

        '''LOGGER'''
        time_end = time.time()
        accumulated_iteration_time.update(time_end - time_start)
        eta_cur_epoch = str(datetime.timedelta(seconds = int(accumulated_iteration_time() * (num_batches - iteration))))        
        if iteration % config['print_every'] == 0:
          print(datetime.datetime.now(pytz.timezone('Asia/Kolkata')).replace(microsecond = 0), end = ' ')

          print('Epoch: %d [%d / %d] ; eta: %s' % (epoch, iteration, num_batches, eta_cur_epoch))
          print('Triplet loss: %f(%f);' % (triplet_loss, accumulated_triplet_loss()))

          wandb.log({'Average Triplet loss': accumulated_triplet_loss()}, step = wandb_step)


          print('Sketch domain loss: %f; Image Domain loss: %f' % (accumulated_sketch_domain_loss(), accumulated_image_domain_loss()))
          wandb.log({'Average Sketch Domain loss': accumulated_sketch_domain_loss()}, step = wandb_step)
          wandb.log({'Average Image Domain loss': accumulated_image_domain_loss()}, step = wandb_step)

          
      '''END OF EPOCH'''
      epoch_end_time = time.time()
      print('Epoch %d complete, time taken: %s' % (epoch, str(datetime.timedelta(seconds = int(epoch_end_time - epoch_start_time)))))
      torch.cuda.empty_cache()

      
#       sketches, image_grids, test_mAP = evaluate(config['test_batch_size'], self.dataloaders.get_test_dataloader, image_model, sketch_model, self.dataloaders.test_dict, k = 5, num_display = 5)
#       wandb.log({'Sketches': [wandb.Image(image) for image in sketches]}, step = wandb_step)
#       wandb.log({'Retrieved Images': [wandb.Image(image) for image in image_grids]}, step = wandb_step)
#       wandb.log({'Average Test mAP': test_mAP}, step = wandb_step)

#       save_checkpoint({'iteration': wandb_step, 
#                         'image_model': image_model.state_dict(), 
#                         'sketch_model': sketch_model.state_dict(),
#                         'optim_dict': optimizer.state_dict()},
#                         checkpoint_dir = 'experiments/', save_to_cloud = True)

      save_checkpoint({'iteration': wandb_step, 
                        'image_model': image_model.state_dict(), 
                        'sketch_model': sketch_model.state_dict(),
                        'domain_net': domain_net.state_dict(),
                        'optim_dict': optimizer.state_dict(),
                        'domain_optim_dict': domain_optim.state_dict()},
                         checkpoint_dir = config['checkpoint_dir'], save_to_cloud = (epoch % config['save_to_cloud_every'] == 0))
      print('Saved epoch to cloud!')
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
