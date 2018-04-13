import torch
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.autograd import Variable

# rewrite in pytorch!


# rewrite in pytorch!
def weighting(image, batch_size, num_classes, weight_type='log'):
    """
    The custom class weighing function. Requires some considerations.
    INPUTS:
    - images: numpy array of shape (batch_size, num_classes, height, width) 
    - batch_size
    - num_classes
    - weight_type(str): 
        -- 'log': 1 / np.log(1.02 + (freq_c / total_freq_c))
        -- 'median': f = Median_freq_c / total_freq_c
    OUTPUTS:
    - class_weights(list): a list of class weights where each index represents each class label
    and the element is the class weight for that label;
    """
    #initialize dictionary with all 0
    
    label_to_frequency = {}
    for i in range(num_classes):
        label_to_frequency[i] = []
    
    #count frequency of each class for images
    
    for n in range(batch_size):
        for i in range(num_classes):
            class_mask = np.equal(image[n, i], 1)
            class_mask = class_mask.astype(np.float32)
            class_frequency = (class_mask.sum())

            if class_frequency != 0.0:
                label_to_frequency[i].append(class_frequency)

    
    #applying weighting function and appending the class weights to class_weights
    
    class_weights = np.zeros((num_classes))
    
    total_frequency = 0
    for frequencies in label_to_frequency.values():
        total_frequency += sum(frequencies)
    
    i = 0
    if weight_type == 'log':
     
        for class_, freq_ in label_to_frequency.items():
            class_weight = 1 / np.log(1.02 + (sum(freq_) / total_frequency))
            class_weights[i] = class_weight
            i += 1
      
    elif weight_type == 'median':
      
        for i, j in label_to_frequency.items():
            #To obtain the median, we got to sort the frequencies
            j = sorted(j) 

            median_freq_c = np.median(j) / sum(j)
            total_freq_c = sum(j) / total_frequency
            median_frequency_balanced = median_freq_c / total_freq_c
            class_weights[i] = median_frequency_balanced
            i += 1
    
    # as first goes background
    class_weights[0] = 0.0 
    # normalize weights:
    class_weights /= class_weights.sum()
    
    return class_weights

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


def show_pics(imgs, col, row):
    fig = plt.figure(figsize=(5*col,5*row))
    for i in range(0, col*row):
        fig.add_subplot(row, col, i + 1)
        if i < len(imgs):
            plt.imshow(imgs[i])
            plt.title('Class ' + str(i+1))
        plt.axis("off")
        
def binarize_classes(classes, num):
    classes_bin = np.zeros((num, classes.shape[0], classes.shape[1]))
    for i in range(len(classes)):
        row = classes[i]
        for j in range(len(row)):
            classes_bin[int(row[j]), i, j] = 1.0
            
    return classes_bin

def show_test_predicts(loader, model):
    inputs, classes = next(iter(loader))
    input_img = torch.unsqueeze(Variable(inputs.cuda(async=True)), dim=0)[0]
    model.cuda()
    model.eval()
    output = model(input_img).data.cpu().numpy()
    show_pics(output[0], 3, 3)
    return output

def show_all_classes(dataset, output):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(np.sum(output[0], axis=0))
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(plt.imread(dataset.dataset.get_curr_pic()[0]))
    plt.axis('off')
