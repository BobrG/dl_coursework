import numpy as np
import torch
from torch.autograd import Variable
from base_train import BaseTrainer

class Trainer(BaseTrainer):
    """ Trainer class

    Note: Inherited from BaseTrainer.
        
    Important - metrics should be representative, so that they should be normalized in [0, 1], to monitor training dynamic.
                same requirement to val_loss;
                
    """
    def __init__(self, model, loss, metrics, data_loader, optimizer, epochs,
                 save_dir, save_freq, gpu, verbosity, identifier='',
                 resume='', valid_data_loader=None, val_loss=None, scheduler=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, epochs,
                                      save_dir, save_freq, verbosity, identifier, resume)
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.scheduler = scheduler
        if val_loss is not None:
            self.val_loss = val_loss
        else:
            self.val_loss = loss
        self.valid = True if self.valid_data_loader else False
        self.gpu = gpu

    def train_epoch(self, epoch):
        """ 
        Train an epoch
        
        """
        self.model.train()
        if self.gpu:
            self.model.cuda()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target) in enumerate(self.data_loader):
            
            if self.gpu:
                data, target = data.cuda(async=True), target.cuda(async=True)
            
            data, target = torch.unsqueeze(Variable(data), dim=0)[0], Variable(target)
           
            self.optimizer.zero_grad()
            output = self.model(data)
           
            for i, metric in enumerate(self.metrics):
                metrics[i] = metric(output, target)
                total_metrics[i] += metrics[i]

            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.data[0]

            step = int(np.sqrt(self.batch_size))
            if self.verbosity >= 2 and batch_idx % step == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss {:.9f}'.format(
                    epoch, batch_idx * len(data), len(self.data_loader) * len(data),
                    100.0 * batch_idx / len(self.data_loader),
                    loss.data[0] ))
                for i in range(len(metrics)):
                    print('Metric ' + self.metrics[i].__name__(), ': {:.9f}'.format(metrics[i]))
            
        avg_metrics = (total_metrics / len(self.data_loader)).tolist()
        avg_loss = total_loss / len(self.data_loader)
        
        log = {'loss': avg_loss, 'metrics': avg_metrics}

        if self.valid:
            val_log = self.valid_epoch()
            log = {**log, **val_log}

        return log

    def valid_epoch(self):
        """
        Validate after training an epoch
        
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target) in enumerate(self.valid_data_loader):
         
            if self.gpu:
                data, target = data.cuda(), target.cuda()
            data, target = torch.unsqueeze(Variable(data), dim=0)[0], Variable(target)
           
            
            output = self.model(data)
            for i, metric in enumerate(self.metrics):
                metrics[i] = metric(output, target)
                total_val_metrics[i] += metrics[i]
                
            loss = self.val_loss(output, target)
            
            total_val_loss += loss.data[0]
        
        avg_metrics = (total_val_metrics / len(self.valid_data_loader)).tolist()
        avg_val_loss = total_val_loss / len(self.valid_data_loader)
        
        if self.scheduler is not None:
            self.scheduler.step(avg_val_loss)
            
        return {'val_loss': avg_val_loss, 'val_metrics': avg_metrics}
