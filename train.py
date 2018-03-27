import numpy as np
import torch
from torch import nn


def train(train_loader, model, loss, optimizer, epoch, gpu=False):
    model.train()
    if gpu == True:
        model.cuda()
    loss_tr = []
    
    for batch_idx, (data, target) in (enumerate(train_loader)):
        
        if gpu == True:
            data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        
        data, target = torch.unsqueeze(torch.Variable(data), dim=0)[0], torch.Variable(target)
       
        optimizer.zero_grad()
        
 
        # Weighted IoU:
        output = model(data)
        num_cl = target.size(1)
        loss_value = 0
        for i in range(num_cl):
            loss_values += loss(output[:, i].resize(target.size(0), 1, target.size(2), target.size(3)),
                                   target[:, i].resize(target.size(0), 1, target.size(2), target.size(3)),
                                   smooth=1e-15,
                                   weight=train_loader.dataset.weights[i])
        loss_value /= num_cl
        
        loss_tr.append(loss_value.data[0])
        loss_value.backward()
       
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: WeightIoU {:.7f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    
    return loss_tr

def validate(val_loader, model, loss, epoch, gpu=False, scheduler=None):
    model.eval()
    if gpu == True:
        model.cuda()
    val_loss = []

    print('Validation', end='')
    
    for batch_idx, (data, target) in (enumerate(val_loader)):
       
        if gpu == True:
            data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
       
        #target, weights = restruct_classes(target, val_loader.dataset.weights, 8)
    
        data, target = torch.unsqueeze(torch.Variable(data), dim=0)[0], torch.Variable(target)
        
        output = model(data)
        num_cl = target.size(1)
        loss_value = 0
        for i in range(num_cl):
            loss_value += loss(output[:, i].resize(target.size(0), 1, target.size(2), target.size(3)),
                                   target[:, i].resize(target.size(0), 1, target.size(2), target.size(3)),
                                   smooth=1e-15,
                                   weight=val_loader.dataset.weights[i])
        loss_value /= num_cl
        
       
        val_loss.append(loss_value.data[0])
        print('.', '')
       
    val_loss.append(np.mean(val_loss))
    if scheduler is not None:
        scheduler.step(val_loss[-1])
        
    print()
    print('Val Epoch: {} Loss on validation: {:.7f}'.format(epoch, val_loss[-1]))
    
    return val_loss[-1]
