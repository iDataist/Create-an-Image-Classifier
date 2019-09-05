import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
from torch import nn, optim
from torchvision import models
from PIL import Image
import json
import argparse
plt.switch_backend('agg')

def load_checkpoint(save_directory, architecture):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(save_directory)
    model = getattr(models, architecture)(pretrained=True)
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
    im.thumbnail((256, 256), Image.ANTIALIAS)
    
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

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    device = 'cpu'
    image = process_image(image_path)  
    image = torch.from_numpy(image) 
    image = image.unsqueeze_(0)
    image = image.float()
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

def view_classify(category_names, image_path, model, topk):
    ''' Function for viewing an image and it's predicted classes.
    '''
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f) 
    probs, classes = predict(image_path, model, topk)
    names = [cat_to_name[c] for c in classes]
    flower = cat_to_name[image_path.split('/')[3]]
    
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), nrows=2)
    ax1.imshow(Image.open(image_path))
    ax1.axis('off')
    ax1.set_title(flower)
    
    ax2.barh(np.arange(topk), np.flip(probs, axis = 0))
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(topk))
    ax2.set_yticklabels(np.flip(names, axis = 0), size='small');
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()
    
def main(save_directory, architecture, image_path, topk, category_names):
    model = load_checkpoint(save_directory, architecture)
    
    image = process_image(image_path)
    imshow(image, ax=None, title=None)
    probs, classes = predict(image_path, model, topk)
      
    view_classify(category_names, image_path, model, topk)
    return probs, classes
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Predict flower name from an image along with the probability of that name.')
    parser.add_argument('--save_directory', default = 'ImageClassifier/checkpoint.pth')
    parser.add_argument('--architecture', default = 'alexnet') 
    parser.add_argument('--image_path', default = 'ImageClassifier/flowers/test/1/image_06743.jpg')
    parser.add_argument('--topk', default = 5, type = int)
    parser.add_argument('--category_names', default = 'ImageClassifier/cat_to_name.json')
    args = parser.parse_args()
    
    main(args.save_directory, args.architecture, args.image_path, args.topk, args.category_names)
