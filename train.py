import numpy as np
import torch
from torch.autograd import Variable
from base_train import BaseTrainer

#WEIGHTENING FUNCTION
def weighting(image, batch_size, num_classes=25):
    """
    The custom class weighing function.
    INPUTS:
    - image_files(list): a list of image_filenames which element can be read immediately;
    OUTPUTS:
    - class_weights(list): a list of class weights where each index represents each class label
    and the element is the class weight for that label;
    """
    #initialize dictionary with all 0
    
    label_to_frequency = {}
    
    for i in range(num_classes):
        label_to_frequency[i] = 0
    
    #count frequency of each class for images
    
    for n in range(batch_size):
#         tmp = image_files[n].split('frame')
#         image = sio.loadmat(tmp[0] + '_segm.mat')['segm_' + str(int(tmp[1][0:-4]) + 1)] 
    
        for i in range(num_classes):
            
            class_mask = np.equal(image[n, i], 1)
            class_mask = class_mask.astype(np.float32)
            class_frequency = (class_mask.sum())

            if class_frequency != 0.0:
                label_to_frequency[i]+=(class_frequency)

    
    #applying weighting function and appending the class weights to class_weights
    
    class_weights = np.zeros((num_classes))
    
    total_frequency = sum(label_to_frequency.values())
    i = 0
    for class_, freq_ in label_to_frequency.items():
        class_weight = 1 / np.log(1.02 + (freq_ / total_frequency))
        class_weights[i] = class_weight
        i += 1
        
    # as first goes background
    class_weights[0] = 0.0 
    
    return class_weights

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
            
            weights = weighting(target.numpy(), target.size(0), target.size(1))
            
           
            if self.gpu:
                data, target = data.cuda(async=True), target.cuda(async=True)
            
            data, target = torch.unsqueeze(Variable(data), dim=0)[0], Variable(target)

           
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss.evaluate(output, target, weights)
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
            weights = weighting(target.numpy(), target.size(0), target.size(1))
            
            if self.gpu:
                data, target = data.cuda(), target.cuda()
            data, target = torch.unsqueeze(Variable(data), dim=0)[0], Variable(target)
           
            
            output = self.model(data)
            loss = self.loss.evaluate(output, target)
            total_val_loss += loss.data[0]

        avg_val_loss = total_val_loss / len(self.valid_data_loader)
        return {'val_loss': avg_val_loss}
