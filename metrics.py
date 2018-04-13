import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from utils import weighting

def get_jaccard(y_pred, y_true):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return (intersection / (union - intersection + epsilon)).mean()

class Metric():
    '''
    Base Class for metrics and losses
    '''
    def __init__(self, name='None', params=None):
        self.name = name
        self.params = params
        self.curr_val = 0.0
    def __name__(self):
        return self.name
    def __params__(self):
        return self.params
    def __call__(self, outputs, targets):
        self.curr_val= value
        return value    

# train losses:

class BCE_LogJac(Metric):
    """
    Multiclass loss for binarized targets
    Defined as BCE - log(soft_jaccard)

    Reference:
        Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
        Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
        arXiv:1706.06169
    """

    def __init__(self, smooth=1e-15, jaccard_weight=1.0):
        super(BCE_LogJac, self).__init__(name='BCE - log(soft_jaccard)', params=['jaccard_weight', 'smooth'])
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight
        self.smooth = smooth

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        batch_size, num_cl = targets.size(0), targets.size(1)
        w, h = targets.size(2), targets.size(3)
        
        for i in range(num_cl):
            out = outputs[:, i]#.resize(batch_size, 1, w, h)
            targ = targets[:, i]#.resize(batch_size, 1, w, h)
            
            jaccard_target = (targ == 1).float()
            jaccard_output = out

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= (self.jaccard_weight / num_cls) * torch.log((intersection + self.smooth) / (union - intersection + self.smooth))
            
        return loss
    
class LossMulti(Metric):
    """
    Multiclass loss for binarized targets
    Defined as soft_jaccard - NegLogLikeLoss

    Reference:
        Shvets, Alexey and Rakhlin, Alexander and Kalinin, Alexandr A and Iglovikov Vladimir,
        Automatic Instrument Segmentation in Robot-Assisted Surgery Using Deep Learning, 2018
    """
    def __init__(self, smooth=1e-15, jaccard_weight=1.0):
        super(LossMulti, self).__init__(name='soft_jaccard - NegLogLikeLoss', params=['jaccard_weight', 'smooth'])
        if class_weights is not None:
            nll_weight = utils.cuda(
                torch.from_numpy(class_weights.astype(np.float32)))
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss2d(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.smooth = smooth

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        batch_size, num_cl = targets.size(0), targets.size(1)
        w, h = targets.size(2), targets.size(3)
        
        if self.jaccard_weight:
            cls_weight = self.jaccard_weight / num_cl
            
            for cls in range(num_cl):
                jaccard_target = (targets[:, cls] == 1.0).float()
                jaccard_output = outputs[:, cls]
                
                intersection = (jaccard_output * jaccard_target).sum()
                union = jaccard_output.sum() + jaccard_target.sum() + self.smooth
                
                loss += (1 - intersection / (union - intersection)) * cls_weight

            loss /= (1 + self.jaccard_weight)
        return loss
    
class Soft_dice_loss(Metric):
    def __init__(self, smooth=1e-15, weighting_type=None):
        super(Soft_dice_loss, self).__init__(name='soft_dice_loss', params=['smooth', 'weights'])
        self.smooth = smooth
        self.curr_val = 0.0
        self.weighting_type = weighting_type
        self.weights = None

    def __call__(self, outputs, targets):
        batch_size, num_cl = targets.size(0), targets.size(1)
        w, h = targets.size(2), targets.size(3)
        value = 0
        
        if self.weighting_type is not None:
            self.weights = weighting(targets.data.cpu().numpy(), targets.size(0), targets.size(1),   weight_type=self.weighting_type)
        else:
            self.weights = np.ones((num_cl)) 
            self.weights[0] = 0.0 # for background

        for i in range(num_cl):
#             out = outputs[:, i].resize(batch_size, 1, w, h)
#             targ = targets[:, i].resize(batch_size, 1, w, h)
            value += (1 - get_jaccard(outputs[:, i], targets[:, i]))*self.weights[i]
        value /= num_cl
        self.curr_val = value
        return value
    def get_weights(self):
        return self.weights
    def get_smooth(self):
        return self.smooth


class Pixel_accuracy_metric(Metric):
    def __init__(self):
        '''
         sum_i(n_ii) / sum_i(t_i), where
         n_ij - the number of pixels of class i
         predicted to belong to class j,
         t_i -  the total number of pixels of class i
        '''
        super(Pixel_accuracy_metric, self).__init__(name='pixel accuracy')
        self.curr_val = 0.0
      
    def __call__(self, outputs, targets):
        num_cl = targets.shape[1]
        batch_size = targets.size(0)
        w = targets.size(2)
        h = targets.size(3)
        value = 0
        n_ii = 0
        t_i = 0      
        
        #count frequency of each class for images
        for i in range(num_cl):
            out = outputs[:, i].resize(batch_size, 1, w, h)
            targ = targets[:, i].resize(batch_size, 1, w, h)
            
            n_ii += (out.view(-1) * targ.view(-1)).sum()
            t_i += targ.view(-1).sum()
       
        value = n_ii / t_i
        
        self.curr_val = value
        return value

class Mean_accuracy_metric(Metric):
    def __init__(self, weighting=None):
        '''
         (1/n_classes)*sum_i(n_ii / t_i), where
         n_ij - the number of pixels of class i
         predicted to belong to class j,
         t_i -  the total number of pixels of class i
         n_classes - the total number of classes in segmentation
        '''
        super(Mean_accuracy_metric, self).__init__(name='mean accuracy')
        self.curr_val = 0.0
      
    def __call__(self, outputs, targets):
        num_cl = targets.shape[1]
        batch_size = targets.size(0)
        w = targets.size(2)
        h = targets.size(3)
        value = 0
        # dictionary contains the number of pixels of class i predicted to belong to class i
        n_ = {}
        for i in range(num_cl):
            n_[i] = 0
        # dictionary contains the total number of pixels of each class
        t_ = {}
        for i in range(num_cl):
            t_[i] = 0
        
        accuracy = list([0]) * num_cl
        #count frequency of each class for images
        
        for n in range(batch_size):
            for i in range(num_cl):

                out = outputs[:, i].resize(batch_size, 1, w, h)
                targ = targets[:, i].resize(batch_size, 1, w, h)
            
                n_ii += (out.view(-1) * targ.view(-1)).sum()
                t_i += targ.view(-1).sum()

                n_[i] += n_ii
                t_[i] += t_i

                if (t_i != 0):
                    accuracy[i] = n_ii / t_i

        value = np.mean(accuracy)
        self.curr_val = value
        return value

class IoU(Metric):
    def __init__(self, smooth=1e-15, weights=None):
        super(IoU, self).__init__(name='intersection over union', params=['smooth', 'weights'])
        self.curr_val = 0.0
        self.smooth = smooth
        if weights is not None:
            self.weights = torch.from_numpy(weights).cuda(async=True)
        else:
            self.weights = torch.from_numpy(np.ones((num_cl))).cuda(async=True)
            self.weights[0] = 0.0
        
    def __call__(self, outputs, targets):
        batch_size, num_cl = targets.size(0), targets.size(1)
        w, h = targets.size(2), targets.size(3)
        value =0
        
        for i in range(num_cl):
            out = outputs[:, i]#.resize(batch_size, 1, w, h)
            targ = targets[:, i]#.resize(batch_size, 1, w, h)
            
            value += get_jaccard(outputs[:, i], targets[:, i]) * self.weights[i]
        
        value /= num_cl
        self.curr_val = value 
        return value

