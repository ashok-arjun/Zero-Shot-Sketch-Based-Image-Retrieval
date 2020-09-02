import torch
from smart_open import open
import os
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
from io import BytesIO
import glob
import torchvision
import torchvision.transforms as T

np.seterr(divide='ignore', invalid='ignore')

def get_data_list(data_dir, labels, label_to_index, section):
  ext = '*.jpg' if section == 'photos' else '*.png'

  filenames = []
  classes = []    
  for label in labels:
    cur_label_filenames = glob.glob(os.path.join(data_dir,section,label,ext))
    filenames.extend(cur_label_filenames)
    classes.extend([label_to_index[label]] * len(cur_label_filenames))

  return filenames, classes    

def get_random_image(image_label_indices, image_filenames, label_idx):
  indices = [i for i,label in enumerate(image_label_indices) if label == label_idx]
  return image_filenames[np.random.choice(indices, 1)[0]]

def label2index(labels):
  '''For saving data when storing the labels'''
  d = {l: i for i, l in enumerate(labels)}
  return d

class SketchyTestDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, labels, label_to_index, embedding, section, transforms = None):
    self.labels = labels
    self.label_to_index = label_to_index
    self.embedding = embedding # not used
    self.transforms = transforms
        
    self.filenames, self.label_idxs = get_data_list(data_dir, self.labels, self.label_to_index, section) 

  def __getitem__(self, idx):
    '''
    Reads image/sketch at 'idx', with its label index
    '''

    '''GET FILENAME AND LABEL'''
    filename = self.filenames[idx]
    label_idx = self.label_idxs[idx]

    image = Image.open(filename).convert('RGB').resize((224,224))

    '''TRANSFORMS'''
    if self.transforms:
      image = self.transforms(image)

    return image, label_idx

  def __len__(self):
    return len(self.filenames)



class SketchyTrainDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, labels, label_to_index, embedding, transforms = None):
    self.labels = labels
    self.label_to_index = label_to_index
    self.embedding = embedding
    self.transforms = transforms

    self.image_filenames, self.image_label_idxs = get_data_list(data_dir, self.labels, self.label_to_index, 'photos') 
    self.sketch_filenames, self.sketch_label_idxs = get_data_list(data_dir, self.labels, self.label_to_index, 'sketches') 

    wv_distance = cdist(embedding, embedding, 'minkowski')
    numerator = np.ones_like(wv_distance); numerator[wv_distance == 0] = 0.0
    self.word_vectors_similarity = numerator/(wv_distance); self.word_vectors_similarity[wv_distance == 0] = 1.0

  def __getitem__(self, idx):
    '''
    Reads sketch at 'idx' and returns a corresponding random image with the same label, and a hard negative image as triplet
    '''
    '''SKETCH IMAGE'''
    sketch_filename, label_idx = self.sketch_filenames[idx], self.sketch_label_idxs[idx]
    sketch_image = Image.open(sketch_filename).convert('RGB').resize((224,224))

    '''POSITIVE IMAGE'''
    positive_image = Image.open(get_random_image(self.image_label_idxs, self.image_filenames, label_idx)).convert('RGB').resize((224,224))  

    '''NEGATIVE IMAGE'''
    # hard negative mining - closer classes(more similarity value) have a higher probability of selection
    current_label_similarities = self.word_vectors_similarity[label_idx]
    negative_labels = [x for x in self.label_to_index.values() if x != label_idx]
    negative_labels_similarities = [current_label_similarities[x] for x in negative_labels]
    negative_labels_similarities_norms = np.linalg.norm(negative_labels_similarities, ord = 1) 
    negative_labels_similarities /= negative_labels_similarities_norms # prob
    chosen_negative_label_idx = np.random.choice(negative_labels, 1, p = negative_labels_similarities)[0]
    negative_image = Image.open(get_random_image(self.image_label_idxs, self.image_filenames, chosen_negative_label_idx)).convert('RGB').resize((224,224)) 


    '''EMBEDDING'''
    cur_label_embedding = self.embedding[label_idx]

    '''TRANSFORMS'''
    if self.transforms:
      sketch_image, positive_image, negative_image = self.transforms(sketch_image), self.transforms(positive_image), self.transforms(negative_image)
      cur_label_embedding = torch.FloatTensor(cur_label_embedding)


    '''RETURN'''

    return sketch_image, positive_image, negative_image, cur_label_embedding, label_idx, chosen_negative_label_idx


  def __len__(self):
    return len(self.sketch_filenames)



class Dataloaders:
  def __init__(self, data_dir):
    self.train_labels = open(os.path.join(data_dir, '../train_labels.txt')).read().splitlines() 
    self.train_dict = label2index(self.train_labels)
    self.train_label_embeddings = np.load(os.path.join(data_dir,'train_embeddings.npy'))

    self.test_labels = open(os.path.join(data_dir, '../test_labels.txt')).read().splitlines() 
    self.test_dict = label2index(self.test_labels)
    self.test_label_embeddings = np.load(os.path.join(data_dir,'test_embeddings.npy')) 
    
    print('Training on: ', self.train_labels)
    print('Testing on: ', self.test_labels)

    self.train_dataset = SketchyTrainDataset(data_dir, self.train_labels, self.train_dict, self.train_label_embeddings, transforms = get_train_transforms())
    self.test_dataset_images = SketchyTestDataset(data_dir, self.test_labels, self.test_dict, self.test_label_embeddings, section='photos', transforms = get_test_transforms())
    self.test_dataset_sketches = SketchyTestDataset(data_dir, self.test_labels, self.test_dict, self.test_label_embeddings, section='sketches', transforms = get_test_transforms())

    self.data_dir = data_dir

  def get_train_dataloader(self, batch_size, shuffle = True):

    train_dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                                  batch_size = batch_size,
                                                  shuffle = shuffle,
                                                  num_workers = 0,
                                                  drop_last = True) # since we use batch normalization, and the last batch's size could be 1
    return train_dataloader


  def get_test_dataloader(self, batch_size, section, shuffle = False):
    if section == 'photos':
      dataset = self.test_dataset_images
    else:
      dataset = self.test_dataset_sketches

    test_dataloader = torch.utils.data.DataLoader(dataset, 
                                                  batch_size = batch_size,
                                                  shuffle = shuffle,
                                                  num_workers = 0)
    return test_dataloader                                              

  def get_full_train_dataloader(self, batch_size, section, shuffle = False):
    dataset = SketchyTestDataset(self.data_dir, self.train_labels, self.train_dict, self.train_label_embeddings, section, transforms = get_test_transforms())

    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size = batch_size,
                                            shuffle = shuffle,
                                            num_workers = 0)
    return dataloader   


def get_train_transforms():
  return T.Compose([T.ToTensor()])


def get_test_transforms():
  return T.Compose([T.ToTensor()])
