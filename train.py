import numpy as np
import torch
from torch.autograd import Variable
from base_train import BaseTrainer


class Trainer(BaseTrainer):
    """ Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, data_loader, optimizer, epochs,
                 save_dir, save_freq, gpu, verbosity, identifier='',
                 resume='', valid_data_loader=None):
        super(Trainer, self).__init__(model, loss, optimizer, epochs,
                                      save_dir, save_freq, verbosity, identifier, resume)
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader else False
        self.gpu = gpu

    def _train_epoch(self, epoch):
        """ 
        Train an epoch
        
        """
        self.model.train()
        if self.gpu:
            self.model.cuda()

        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.data_loader):
            
            if self.gpu:
                data, target = data.cuda(async=True), target.cuda(async=True)
            
            data, target = torch.unsqueeze(Variable(data), dim=0)[0], Variable(target)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss.evaluate(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.data[0]
            step = int(np.sqrt(self.batch_size))
            if self.verbosity >= 2 and batch_idx % step == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.data_loader) * len(data),
                    100.0 * batch_idx / len(self.data_loader), loss.data[0]))

        avg_loss = total_loss / len(self.data_loader)
        log = {'loss': avg_loss}

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log

    def _valid_epoch(self):
        """
        Validate after training an epoch
        
        """
        self.model.eval()
        total_val_loss = 0
        for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            data, target = torch.unsqueeze(Variable(data), dim=0)[0], Variable(target)
            if self.gpu:
                data, target = data.cuda(), target.cuda()

            output = self.model(data)
            loss = self.loss(output, target)
            total_val_loss += loss.data[0]

        avg_val_loss = total_val_loss / len(self.valid_data_loader)
        return {'val_loss': avg_val_loss}

