class RunningAverage():
  def __init__(self):
    self.count = 0
    self.sum = 0

  def update(self, value, n_items = 1):
    self.sum += value * n_items
    self.count += n_items

  def __call__(self):
    return self.sum/self.count  


def save_checkpoint(state, is_best, checkpoint_dir, train = True):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    prefix = 'train' if train else 'test'
    torch.save(state, os.path.join(checkpoint_dir, prefix + '_last.pth.tar'))
    # torch.save(state, os.path.join(wandb.run.dir, prefix + "_last.pth.tar"))
#     wandb.save(prefix + '_last.pth.tar')
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, prefix + '_best.pth.tar'))
        # torch.save(state, os.path.join(wandb.run.dir, prefix + "_best.pth.tar"))
#         wandb.save(prefix + '_best.pth.tar')

def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def save_epoch_to_cloud(state, epoch_index):
    prefix = str(epoch_index)
    print('Trying to save epoch', prefix)
    torch.save(state, os.path.join(wandb.run.dir, 'epoch_' + prefix + ".pth.tar"))
    print('Epoch saved to cloud: ', prefix)
    wandb.save('epoch_' + prefix + ".pth.tar")