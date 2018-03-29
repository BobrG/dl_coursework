import torch

class Loss():
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.curr_val = 0.0
    def __name__(self):
        return self.name
    def __params__(self):
        return self.params
    def evaluate(self, outputs, targets):
        self.curr_val= value
        return value
    
    
class Soft_dice_loss(Loss):
    def __init__(self, smooth=1e-10):#, weights=1.0):
        super(Soft_dice_loss, self).__init__('soft_dice_loss', ['smooth', 'weights'])
        self.smooth = smooth
        #self.weights = weights
        self.curr_val = 0.0
        
    def evaluate(self, outputs, targets, weights):
        num_cl = targets.shape[1]
        batch_size = targets.size(0)
        w = targets.size(2)
        h = targets.size(3)
        value = 0
        
        for i in range(num_cl):
            out = outputs[:, i].resize(batch_size, 1, w, h)
            targ = targets[:, i].resize(batch_size, 1, w, h)
            
            iflat = out.view(-1)
            tflat = targ.view(-1)
            intersection = (iflat * tflat).sum()
            value += (1 - ((2. * intersection + self.smooth) /
                  (iflat.sum() + tflat.sum() + self.smooth)))*weights[i]
        value = value / num_cl
        self.curr_val = value
        return value
    def change_weights(self, new_weights):
        self.weights = new_weights