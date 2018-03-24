from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from metrics import soft_dice_loss



def train(train_loader, model, epoch, gpu=False):
    model.train()
    if gpu == True:
        model.cuda()
    loss_tr = []
    
    for batch_idx, (data, target) in (enumerate(train_loader)):
    
        if gpu == True:
            data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        
        data, target = torch.unsqueeze(Variable(data), dim=0)[0], Variable(target)
       
        optimizer.zero_grad()
        
# Simple summ of losses:
#         loss_1 = F.binary_cross_entropy(model(data), target)
#         output = model(data)

#         loss_2 = soft_dice_loss(output, target)
#         loss = loss_1 + loss_2
#         loss_tr.append([loss_1.data[0], loss_2.data[0]])

 
# Weighted IoU:
        output = model(data)
        num_cl = target.size(1)
        loss = 0
        for i in range(num_cl):
            loss += soft_dice_loss(output[:, i].resize(target.size(0), 1, 256, 320),
                                   target[:, i].resize(target.size(0), 1, 256, 320),
                                   smooth=1e-15,
                                   weight=train_w[i+1])
        loss /= num_cl
        
        loss_tr.append(loss.data[0])
        loss.backward()
       
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: WeightIoU {:.7f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    
    return loss_tr

def validate(val_loader, model, epoch, gpu=False, scheduler=None):
    model.eval()
    if gpu == True:
        model.cuda()
    val_loss = []
    
    for batch_idx, (data, target) in (enumerate(val_loader)):
       
        if gpu == True:
            data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
       
        data, target = torch.unsqueeze(Variable(data), dim=0)[0], Variable(target)
        
        optimizer.zero_grad()
        
        output = model(data)
        num_cl = target.size(1)
        loss = 0
        for i in range(num_cl):
            loss += soft_dice_loss(output[:, i].resize(target.size(0), 1, 256, 320),
                                   target[:, i].resize(target.size(0), 1, 256, 320),
                                   smooth=1e-15,
                                   weight=val_w[i+1])
        loss /= num_cl
        
       
        val_loss.append(loss.data[0])
       
    val_loss.append(np.mean(val_loss))
    if scheduler is not None:
        scheduler.step(val_loss[-1])
        
    print('Val Epoch: {} Loss on validation: {:.7f}'.format(epoch, val_loss[-1]))
    
    return val_loss[-1]
