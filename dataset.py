import cv2
import PIL

#USEFULL FUNCTION
def load_image(path, pad=True):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (network requirement)
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if not pad:
        return img
    
    height, width, _ = img.shape
    
    if height % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - height % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad
        
    if width % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - width % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad
    
    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


#WEIGHTENING FUNCTION
def weighting(image, num_classes=25):
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
    
    for n in range(len(image)):
#         tmp = image_files[n].split('frame')
#         image = sio.loadmat(tmp[0] + '_segm.mat')['segm_' + str(int(tmp[1][0:-4]) + 1)] 
    
        for i in range(num_classes - 1):
            
            class_mask = np.equal(image[:, i], 1)
            #class_mask = class_mask.astype(np.float32)
            class_frequency = (class_mask.sum())

            if class_frequency != 0.0:
                label_to_frequency[i]+=(class_frequency)

    
    #applyinf weighting function and appending the class weights to class_weights
    
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

#SURREAL DATASET LOADER
class SURREALDataset(Dataset):
    def __init__(self, dirry, transforms=None, lengt=None, weights=None):
        self.pics = []
        
        subdirs = [i[0] for i in os.walk(dirry)]
        
        for i in subdirs:
            for j in [k for k in os.listdir(i) if k.endswith('jpg')]:
                self.pics.append(i + '/' + j)
                if (len(self.pics) > lengt and lengt is not None):
                    break
        
        self.transforms = transforms
        if weights is None:
            self.weights = weighting(self.pics, 25)
        else:
            self.weights = weights
            
    def __getitem__(self, i):
        
        num_classes = 25
        
        # get image and add padding to shape (x_height // 32 == 0, x_width // 32 == 0)
        x, pad = load_image(self.pics[i], pad=True)
    
        # get segmentation map
        tmp = self.pics[i].split('frame')
        mask = sio.loadmat(tmp[0] + '_segm.mat')['segm_' + str(int(tmp[1][0:-4]) + 1)] 
        
#         mask[(mask == 2) & (mask == 3)] = 1
#         mask[(mask == 4) &
#              (mask == 7) &
#              (mask == 10) &
#              (mask == 13) &
#              (mask == 14) &
#              (mask == 15)] = 2
#         mask[(mask == 5) & (mask == 6)] = 3
#         mask[(mask == 8) &
#              (mask == 9) &
#              (mask == 11) &
#              (mask == 12)] = 4
#         mask[mask == 16] = 5
#         mask[(mask == 17) & (mask == 18)] = 6
#         mask[(mask == 19) & (mask == 20)] = 7
#         mask[(mask == 21) & 
#              (mask == 22) &
#              (mask == 23) &
#              (mask == 24)] = 8
        
        y = np.zeros((num_classes, x.shape[0], x.shape[1]))
        
        #y = binarize_classes(mask, 8) 
        
        for i in range(len(mask)):
            row = mask[i]
            for j in range(len(row)):
                y[row[j], i, j] = 1.0

        # transorm ToTensor + add normalization to zero mean and unit std
        if self.transforms is not None:
            x = self.transforms(x)
        for i in range(0, 3):
            x[i] -= x[i].mean()
            
            x[i] /= x[i].std()
            
        # get all maps or only one
        y = torch.FloatTensor(y[:6]) #np.rollaxis(y, 0, 3)
       
        return x, y
    
    def __len__(self):
        return len(self.pics)
    
    def get_pics_path(self, indx):
        return self.pics[indx]

#SITTING PEOPLE DATASET LOADER    
class SittingDataset(Dataset):
    def __init__(self, dirry, transforms=None, nclasses = True):
        self.pics = []
        for j in [k for k in os.listdir(dirry) if k.endswith('jpg')]:
            self.pics.append(dirry + '/' + j)

        self.classes = nclasses
        self.transforms = transforms
    def __getitem__(self, i):
        # get image 
        x, pad = load_image(self.pics[i], pad=True)
       
        # get segmentation map
        tmp = self.pics[i].split('img')
        mask = sio.loadmat(tmp[0] + 'masks' + tmp[-1][:-4] + '.mat')['M']   
        y = np.zeros((25, 320, 320))
        
        
        for i in range(len(mask)):
            row = mask[i]
            for j in range(len(row)):
                y[int(row[j]), i, j] = 1.0

        if self.transforms is not None:
            x = self.transforms(x)
       
        for i in range(0, 3):
            x[i] -= x[i].mean()
            
            x[i] /= x[i].std()
    
        if self.classes:
            y = torch.FloatTensor(y[1:]) 
        else:
            y = torch.FloatTensor(y[1]) 
        return x, y
    
    def __len__(self):
        return len(self.pics)

