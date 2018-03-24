import torch

def soft_dice_loss(outputs, targets, smooth=1e-15, weight = 1.0):
    iflat = outputs.view(-1)
    tflat = targets.view(-1)
    intersection = (iflat * tflat).sum()
    
    return (1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth)))*weight
