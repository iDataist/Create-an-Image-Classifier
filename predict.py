import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from time import time
from PIL import Image
import json
%config InlineBackend.figure_format = 'retina'

"""
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
"""  

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
def load_checkpoint(save_directory):
    checkpoint = torch.load(save_directory)
    model = models.alexnet(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image_path)
 
    # Resize the image
    im = im.resize((256, 256))
    
    # Crop out the center 224x224 portion of the image
    width, height = im.size 
    new_width, new_height = 224, 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    im = im.crop((left, top, right, bottom))    
    
    # Convert image to numpy array
    np_image = np.array(im)/255
    
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    return np_image.transpose((2,0,1))

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    image = process_image(image_path)  
    image = torch.from_numpy(image) 
    image = image.unsqueeze_(0)
    image = image.to(device)
    
    model.to(device)       
    model.eval()
    
    with torch.no_grad():  
        logps = model.forward(image)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim = 1)
    top_p, top_class = np.array(top_p)[0], np.array(top_class)[0]

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_class = [idx_to_class[i] for i in top_class]
    return top_p, top_class

def view_classify(image_path, model):
    ''' Function for viewing an image and it's predicted classes.
    '''
    probs, classes = predict(image_path, model, topk=5)
    names = [cat_to_name[c] for c in classes]
    flower = cat_to_name[image_path.split('/')[2]]
    
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), nrows=2)
    ax1.imshow(Image.open(image_path))
    ax1.axis('off')
    ax1.set_title(flower)
    
    ax2.barh(np.arange(5), np.flip(probs, axis = 0))
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels(np.flip(names, axis = 0), size='small');
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
