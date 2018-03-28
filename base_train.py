import os
import math
import shutil
import torch
#from utils.util import ensure_dir


class BaseTrainer:
    """ 
        Base class for all trainer.
        Each train class inherits from this one:)
    """
    def __init__(self, model, loss, optimizer, epochs,
                 save_dir, save_freq, verbosity, identifier='', resume=''):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.verbosity = verbosity
        self.identifier = identifier
        self.min_loss = math.inf
        self.start_epoch = 1
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if resume:
            self._resume_checkpoint(resume)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs+1):
            result = self._train_epoch(epoch)
            if verbosity >= 1:
                print(result)
            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch, result['loss'])

    def _train_epoch(epoch):
        print('Not implemented yet')
            
    def _save_checkpoint(self, epoch, loss):
        if loss < self.min_loss:
            self.min_loss = loss
        arch = type(self.model).__name__
        state = {
            'epoch': epoch,
            'arch': arch,
            'loss': loss.__name__,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'min_loss': self.min_loss,
        }
        filename = os.path.join(self.save_dir,
                                self.identifier + 'checkpoint_epoch{:02d}_loss_{:.5f}.pth.tar'.format(epoch, loss))
        print("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)
        if loss == self.min_loss:
            shutil.copyfile(filename, os.path.join(self.save_dir, 'model_best.pth.tar'))

    def _resume_checkpoint(self, resume_path):
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.min_loss = checkpoint['min_loss']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger = checkpoint['logger']
        print("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
