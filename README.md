# Create an Image Classifier

## Project Overview
Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. The goal of this project is to develop code for an image classifier built with PyTorch, and then convert it into a command line application.

## Key Skills Demonstrated
- PyTorch and neural networks
- Model validation and evaluation

## Data

We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below.

<img src='assets/Flowers.png' width=500px>

## Project steps

- Develop code for an image classifier built with PyTorch. The code development process is documented in [Image Classifier Project.ipynb](https://github.com/iDataist/Create-an-Image-Classifier/blob/master/Image%20Classifier%20Project.ipynb).
  - Load and preprocess the image dataset
  - Train the image classifier on the dataset
  - Use the trained classifier to predict image content

- Convert the image classifier into a command line application. The command line application includes two Python scripts, [train.py](https://github.com/iDataist/Create-an-Image-Classifier/blob/master/train.py) and [predict.py](https://github.com/iDataist/Create-an-Image-Classifier/blob/master/predict.py), that run from the command line. The first file, **train.py**, trains a new network on a dataset and save the model as a checkpoint. The second file, **predict.py**, uses a trained network to predict the class for an input image.

- **Train a new network on a data set with train.py**

  - Basic usage: ```python train.py --data_dir /path/to/images```

  - Set directory to save checkpoints: ```python train.py --save_directory /path/to/checkpoint```

  - Choose architecture: ```python train.py --architecture "alexnet"```

  - Set hyperparameters: ```python train.py --hidden_units 2014 512 --learning_rate 0.001 --epochs 3```

  - Use GPU for training: ```python train.py --gpu```

- **Predict flower name from an image with predict.py along with the probability of that name**

  - Basic usage: ```python predict.py --image_path /path/to/image --save_directory /path/to/checkpoint```

  - Return top K most likely classes: ```python predict.py --topk 3```

  - Use a mapping of categories to real names: ```python predict.py --category_names cat_to_name.json```

  - Use GPU for inference: ```python predict.py --gpu```

## About Deep Learning
