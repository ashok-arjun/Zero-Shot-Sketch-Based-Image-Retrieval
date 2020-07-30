import torch
from smart_open import open
import os
from zipfile import ZipFile
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
from io import BytesIO
import torchvision
import torchvision.transforms as T

def get_data_list(zip_file, labels, label_to_index, data_type):
  if data_type == 'images':
    ext = '.jpg'
    prefix = 'QuickDraw_images_final'
  elif data_type == 'sketches': 
    ext = '.png'
    prefix = 'QuickDraw_sketches_final'

  filenames = []
  classes = []    
  for label in labels:
    cur_label_filenames = [fn for fn in zip_file if fn.startswith(os.path.join(prefix, label)) and fn.endswith(ext)]
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

class QuickDrawTestDataset(torch.utils.data.Dataset):
  def __init__(self, zip_file, labels, label_to_index, embedding, section, transforms = None):
    self.zip_file = zip_file
    self.labels = labels
    self.label_to_index = label_to_index
    self.embedding = embedding # not used
    self.transforms = transforms

    self.filenames, self.label_idxs = get_data_list(self.zip_file, self.labels, self.label_to_index, section) 

  def __getitem__(self, idx):
    '''
    Reads image/sketch at 'idx', with its label index
    '''

    '''GET FILENAME AND LABEL'''
    filename = self.filenames[idx]
    label_idx = self.label_idxs[idx]

    image = Image.open(BytesIO(self.zip_file[filename])).convert('RGB').resize((224,224))

    '''TRANSFORMS'''
    if self.transforms:
      image = self.transforms(image)

    return image, label_idx

  def __len__(self):
    return len(self.filenames)



class QuickDrawTrainDataset(torch.utils.data.Dataset):
  def __init__(self, images_zip_file, sketches_zip_file, labels, label_to_index, embedding, transforms = None):
    self.images_zip_file = images_zip_file
    self.sketches_zip_file = sketches_zip_file
    self.labels = labels
    self.label_to_index = label_to_index
    self.embedding = embedding
    self.transforms = transforms

    self.image_filenames, self.image_label_idxs = get_data_list(self.images_zip_file, self.labels, self.label_to_index, 'images') 
    self.sketch_filenames, self.sketch_label_idxs = get_data_list(self.sketches_zip_file, self.labels, self.label_to_index, 'sketches') 

    self.word_vectors_similarity = np.exp(-np.square(cdist(self.embedding, self.embedding, metric = 'euclidean'))/0.1) # 0.1 is temperature
    # can be cosine similarity also

  def __getitem__(self, idx):
    '''
    Reads sketch at 'idx' and returns a corresponding random image with the same label, and a hard negative image as triplet
    '''

    '''GET FILENAME AND LABEL'''
    filename = self.sketch_filenames[idx]
    label_idx = self.sketch_label_idxs[idx]

    '''SKETCH IMAGE'''
    sketch_image = Image.open(BytesIO(self.sketches_zip_file[filename])).convert('RGB').resize((224,224))

    '''POSITIVE IMAGE'''
    positive_image = Image.open(BytesIO(self.images_zip_file[get_random_image(self.image_label_idxs, self.image_filenames, label_idx = label_idx)])).convert('RGB').resize((224,224))  

    '''NEGATIVE IMAGE'''
    current_label_similarities = self.word_vectors_similarity[label_idx]
    negative_labels = [x for x in self.label_to_index.values() if x != label_idx]
    negative_labels_similarities = [current_label_similarities[x] for x in negative_labels]
    negative_labels_similarities_norms = np.linalg.norm(negative_labels_similarities, ord = 1) # Returns sum over absolute values
    negative_labels_similarities /= negative_labels_similarities_norms
    chosen_negative_label_idx = np.random.choice(negative_labels, 1, p = negative_labels_similarities)[0]
    negative_image = Image.open(BytesIO(self.images_zip_file[get_random_image(self.image_label_idxs, self.image_filenames, label_idx = chosen_negative_label_idx)])).convert('RGB').resize((224,224)) 


    '''EMBEDDING'''
    cur_label_embedding = self.embedding[label_idx]

    '''TRANSFORMS'''
    if self.transforms:
      sketch_image, positive_image, negative_image = self.transforms(sketch_image), self.transforms(positive_image), self.transforms(negative_image)
      cur_label_embedding = torch.FloatTensor(cur_label_embedding)


    '''TRIPLET'''

    triplet = {'anchor':sketch_image, 'positive':positive_image, 'negative':negative_image}

    return triplet, cur_label_embedding, label_idx, chosen_negative_label_idx


  def __len__(self):
    return len(self.sketch_filenames)



class Dataloaders:
  def __init__(self, data_dir = 'data'):
    print('Reading the zip files...')
    self.sketches_zip = self.get_zip_file(os.path.join(data_dir, 'QuickDraw_sketches_final.zip'))
    self.images_zip = self.get_zip_file(os.path.join(data_dir, 'QuickDraw_images_final.zip'))
    print('Done.')


    self.train_labels = open(os.path.join(data_dir, 'train_labels.txt')).read().splitlines() 
    self.train_dict = label2index(self.train_labels)
    self.train_label_embeddings = np.load(os.path.join(data_dir,'train_embeddings.npy'))

    self.test_labels = open(os.path.join(data_dir, 'test_labels.txt')).read().splitlines() 
    self.test_dict = label2index(self.test_labels)
    self.test_label_embeddings = np.load(os.path.join(data_dir,'test_embeddings.npy'))  

  def get_train_dataloader(self, batch_size, shuffle = True):

    train_dataset = QuickDrawTrainDataset(self.images_zip, self.sketches_zip, self.train_labels, self.train_dict, self.train_label_embeddings, transforms = get_train_transforms())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                  batch_size = batch_size,
                                                  shuffle = shuffle,
                                                  num_workers = 4) 
    return train_dataloader, self.train_dict


  def get_test_dataloader(self, batch_size, section, shuffle = False):
    zipfile = self.images_zip if section == 'images' else self.sketches_zip
    test_dataset = QuickDrawTestDataset(zipfile, self.test_labels, self.test_dict, self.test_label_embeddings, section=section, transforms = get_test_transforms())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                  batch_size = batch_size,
                                                  shuffle = shuffle,
                                                  num_workers = 1)
    return test_dataloader, self.test_dict                                              


  def get_zip_file(self, path):
    input_zip = ZipFile(path)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    return data   



def get_train_transforms():
  return T.Compose([T.ToTensor()])


def get_test_transforms():
  return T.Compose([T.ToTensor()])