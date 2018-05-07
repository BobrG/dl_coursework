import os
import cv2
import torch
import numpy as np
import scipy.io as sio
from utils import load_image
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset

#SURREAL DATASET LOADER
class SURREALDataset(Dataset):
    def __init__(self, dirry, num_classes, transforms=None, fixed_size=None, identifier=None, lengt=None, weights=None, weight_type=None):
        self.pics = []
        
        subdirs = [i[0] for i in os.walk(dirry)]
        
        for i in subdirs:
            for j in [k for k in os.listdir(i) if k.endswith('jpg')]:
                self.pics.append(i + '/' + j)
                if (len(self.pics) > lengt and lengt is not None):
                    break
        
        self.transforms = transforms
        self.classes = num_classes
        self.identifier = identifier 
        self.curr_pic = []
        self.curr_mask = []
        self.fixed_size = fixed_size

    def __getitem__(self, i):
        # get image and add padding to shape (x_height // 32 == 0, x_width // 32 == 0)
        if self.fixed_size is not None:
            x, pad = load_image(self.pics[i], pad=True, fixed_size=self.fixed_size)
        else:    
            x, pad = load_image(self.pics[i], pad=True)
        # save current batch of pictures for further demonstration
        if len(self.curr_pic) >= 4:
            self.curr_pic = []
            self.curr_mask = []
           
        self.curr_pic.append(self.pics[i])
        
        # get segmentation map
        tmp = self.pics[i].split('frame')
        mask = sio.loadmat(tmp[0] + '_segm.mat')['segm_' + str(int(tmp[1][0:-4]) + 1)] 
        
        y = np.zeros((self.classes, x.shape[0], x.shape[1]))
        
        #restructing classes if necessary
        if self.identifier == 'restructed':
            mask[(mask == 2) + (mask == 3)] = 1
            mask[(mask == 4) + (mask == 7) +
                 (mask == 10) + (mask == 13) +
                 (mask == 14) + (mask == 15)] = 2
            mask[(mask == 5) + (mask == 6)] = 3
            mask[(mask == 8) + (mask == 9) +
                 (mask == 11) + (mask == 12)] = 4
            mask[mask == 16] = 5
            mask[(mask == 19) + (mask == 20)] = 6
            mask[(mask == 17) + (mask == 18)] = 7
            mask[(mask == 21) + (mask == 22) +
                 (mask == 23) + (mask == 24)] = 8            
        
        
        self.curr_mask.append(mask)
        
        # transorm ToTensor + add normalization to zero mean and unit std
        if self.transforms is not None:
            x, mask = self.transforms(x, mask)
            
        #x = torch.from_numpy(np.moveaxis(x, -1, 0)).float()
        #binarizing mask
        for i in range(len(mask)):
            row = mask[i]
            for j in range(len(row)):
                y[row[j], i, j] = 1.0

        for i in range(0, 3):
            x[i] -= x[i].mean()
            
            x[i] /= x[i].std()
            
        
        y = torch.FloatTensor(y)#drop background
       
        return x, y
    
    def __len__(self):
        return len(self.pics)
    
    def get_pics_path(self, indx):
        return self.pics[indx]
    
    def get_classes(self):
        return self.classes
    
    def get_curr_pic(self):
        return self.curr_pic
    
    def get_curr_mask(self):
        return self.curr_mask
    
#SITTING PEOPLE DATASET LOADER    
class SittingDataset(Dataset):
    def __init__(self, dirry, num_classes, transforms=None, fixed_size=None, identifier=None, lengt=None):
        self.pics = []
        subdirs = [k for k in os.listdir(dirry) if k.endswith('jpg')]
        for j in subdirs:
            self.pics.append(dirry + '/' + j)
        

        self.classes = num_classes
        self.transforms = transforms
        self.identifier = identifier 
        self.curr_pic = []
        self.curr_mask = []
        self.fixed_size = fixed_size

    def __getitem__(self, i):
        # get image 
        if self.fixed_size is not None:
            x, pad = load_image(self.pics[i], pad=True, fixed_size=self.fixed_size)
        else:    
            x, pad = load_image(self.pics[i], pad=True)
       
        if len(self.curr_pic) >= 4:
            self.curr_pic = []
            self.curr_mask = []
           
        self.curr_pic.append(self.pics[i])
        
        # get segmentation map
        tmp = self.pics[i].split('img')
        mask = sio.loadmat(tmp[0] + 'masks' + tmp[-1][:-4] + '.mat')['M']   
        y = np.zeros((self.classes, x.shape[0], x.shape[1]))
        
        # change segmentation maps
        if self.identifier == 'restructed':
            
            mask[(mask == 5) + (mask == 8)] = 8
            mask[(mask == 4) + (mask == 7)] = 7
            mask[(mask == 3) + (mask == 6)] = 6
            mask[(mask == 1)] = 5
            mask[(mask == 2)] = 2
            mask[(mask == 10) + (mask == 13)] = 1
            mask[(mask == 9) + (mask == 12)] = 3  
            mask[(mask == 11) + (mask == 14)] = 4
        

        # transorm ToTensor + add normalization to zero mean and unit std
        if self.transforms is not None:
            x, mask = self.transforms(x, mask)

        self.curr_mask.append(mask)   
        #x = torch.from_numpy(np.moveaxis(x, -1, 0)).float()
        #binarizing mask
        for i in range(len(mask)):
            row = mask[i]
            for j in range(len(row)):
                y[int(row[j]), i, j] = 1.0

       
        for i in range(0, 3):
            x[i] -= x[i].mean()
            
            x[i] /= x[i].std()
    
        y = torch.FloatTensor(y)#drop background
    
        return x, y
    
    def __len__(self):
        return len(self.pics)

    def get_classes(self):
        return self.classes
    
    def get_curr_pic(self):
        return self.curr_pic
    
    def get_curr_mask(self):
        return self.curr_mask

#PASCAL PARTS DATASET LOADER
class PascalDataset(Dataset):
    def __init__(self, dirry, num_classes, transforms=None, fixed_size=None, identifier=None, lengt=None):
        self.pics = []
        self.masks = []
        subdirs = [k for k in os.listdir(dirry) if k.endswith('mat')]
        self.path_2_pic = '/home/novikov/home/novikov/Dropbox/skoltech/students/Gleb/VOCdevkit/VOC2010/JPEGImages'
        self.path_2_mask = dirry
        for j in subdirs:
            self.pics.append(self.path_2_pic + '/' + j[:-4] + '.jpg')
            self.masks.append(self.path_2_mask + '/' + j)
        self.classes = num_classes
        self.transforms = transforms
        self.identifier = identifier 
        self.curr_pic = []
        self.curr_mask = []
        # classes for futher reconstruction
        self.classes_list = [
            ['hair', 'head', 'lear', 'leye', 'lebrow', 'rear', 'reye', 'rebrow', 'nose'],
            ['torso', 'neck'],
            ['lhand', 'rhand'],
            ['ruarm', 'luarm'],
            ['rlarm', 'llarm'],
            ['rlleg', 'llleg'],
            ['ruleg', 'luleg'],
            ['rfoot', 'lfoot']
            ]
        self.fixed_size = fixed_size

    def  __getitem__(self, i):
        if self.fixed_size is not None:
            x, pad = load_image(self.pics[i], pad=True, fixed_size=self.fixed_size)
        else:    
            x, pad = load_image(self.pics[i], pad=True)
            
        if len(self.curr_pic) >= 4:
            self.curr_pic = []
            self.curr_mask = []
        
        self.curr_pic.append(self.pics[i])
        
        # get segmentation map
        y = np.zeros((self.classes, x.shape[0], x.shape[1]))
        tmp_mask=sio.loadmat(self.masks[i])['anno'][0][0][1][0]
        for i in tmp_mask:
            if (i[0][0] == 'person'):
                #background
                #y[0] = cv2.copyMakeBorder(-1*(i[2][0] - np.ones(i[2][0].shape)), pad, cv2.BORDER_REFLECT_101) 
                for j, mask in enumerate(i[3][0][0]):
                    for c, class_ in enumerate(self.classes_list):
                        if mask[0][0] in class_:
                            y[c] = cv2.copyMakeBorder(mask[1], pad, cv2.BORDER_REFLECT_101)
                break

        if self.transforms is not None:
            x, y = self.transforms(x, y)
        y = torch.FloatTensor(y)

        return x, y
    
    def __len__(self):
        return len(self.pics)

    def get_classes(self):
        return self.classes
    
    def get_curr_pic(self):
        return self.curr_pic
    
    def get_curr_mask(self):
        return self.curr_mask


#Combination Dataset
class CombineDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengts = [len(i) for i in datasets]
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)
    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if (index < offset and i > 0):
                index -= self.offsets[i-1]
                return self.datasets[i][index]
    def __len__(self):
        return self.length

#TEMPORARY Sur + Real dataset
class Sur_and_Real(Dataset):
    def __init__(self, dirry, num_classes, transforms=None, identifier=None, lengt=None, ind_1=None, ind_2=None):
        self.pics = []

        subdirs = [k for k in os.listdir(dirry[1]) if k.endswith('jpg')]
        
        for j in range(ind_1, ind_2):
            self.pics.append(dirry[1] + '/' + subdirs[j])
        
        subdirs = [i[0] for i in os.walk(dirry[0])]

        for i in subdirs:
            for j in [k for k in os.listdir(i) if k.endswith('jpg')]:
                
                if (len(self.pics) > lengt and lengt is not None):
                    break
                else:
                    self.pics.append(i + '/' + j)

        self.classes = num_classes
        self.transforms = transforms
        self.identifier = identifier 
        self.curr_pic = []
        self.curr_mask = []

    def __getitem__(self, i):
               
        # get image and add padding to shape (x_height // 32 == 0, x_width // 32 == 0)
        x, pad = load_image(self.pics[i], pad=True)
        #x = cv2.copyMakeBorder(plt.imread(self.pics[i]), 10, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

        # save current batch of pictures for further demonstration
        if len(self.curr_pic) >= 4:
            self.curr_pic = []
            self.curr_mask = []
           
        self.curr_pic.append(self.pics[i])
        
        y = np.zeros((self.classes, x.shape[0], x.shape[1]))

        if 'sur' in self.pics[i]: 
            tmp = self.pics[i].split('frame')
            mask = sio.loadmat(tmp[0] + '_segm.mat')['segm_' + str(int(tmp[1][0:-4]) + 1)] 
            
            if self.identifier == 'restructed':
                mask[(mask == 2) + (mask == 3)] = 1
                mask[(mask == 4) + (mask == 7) +
                    (mask == 10) + (mask == 13) +
                    (mask == 14) + (mask == 15)] = 2
                mask[(mask == 5) + (mask == 6)] = 3
                mask[(mask == 8) + (mask == 9) +
                    (mask == 11) + (mask == 12)] = 4
                mask[mask == 16] = 5
                mask[(mask == 19) + (mask == 20)] = 6
                mask[(mask == 17) + (mask == 18)] = 7
                mask[(mask == 21) + (mask == 22) +
                    (mask == 23) + (mask == 24)] = 8            
        elif 'sit' in self.pics[i]:
            tmp = self.pics[i].split('img')
            mask = sio.loadmat(tmp[0] + 'masks' + tmp[-1][:-4] + '.mat')['M']   

            if self.identifier == 'restructed':
            
                mask[(mask == 5) + (mask == 8)] = 8
                mask[(mask == 4) + (mask == 7)] = 7
                mask[(mask == 3) + (mask == 6)] = 6
                mask[(mask == 1)] = 5
                mask[(mask == 2)] = 2
                mask[(mask == 10) + (mask == 13)] = 1
                mask[(mask == 9) + (mask == 12)] = 3  
                mask[(mask == 11) + (mask == 14)] = 4
                
        
        if self.transforms is not None:
            x, mask = self.transforms(x, mask)

        self.curr_mask.append(mask)   
        #x = torch.from_numpy(np.moveaxis(x, -1, 0)).float()

        #binarizing mask
        for i in range(len(mask)):
            row = mask[i]
            for j in range(len(row)):
                y[int(row[j]), i, j] = 1.0

       
        for i in range(0, 3):
            x[i] -= x[i].mean()
            
            x[i] /= x[i].std()
            
        
        y = torch.FloatTensor(y)#drop background
        
        return x, y
    
    def __len__(self):
        return len(self.pics)

    def get_classes(self):
        return self.classes
    
    def get_curr_pic(self):
        return self.curr_pic
    
    def get_curr_mask(self):
        return self.curr_mask