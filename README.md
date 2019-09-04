# Create an Image Classifier

The goal of this project is to develop code for an image classifier built with PyTorch, and then convert it into a command line application. The code development process is documented in [Image Classifier Project.ipynb](https://github.com/iDataist/Create-an-Image-Classifier/blob/master/Image%20Classifier%20Project.ipynb). The command line application includes two Python scripts,  [train.py](https://github.com/iDataist/Create-an-Image-Classifier/blob/master/train.py) and [predict.py](https://github.com/iDataist/Create-an-Image-Classifier/blob/master/predict.py), that run from the command line. The first file, train.py, trains a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. Below is the command line manual for different tasks.

- Train a new network on a data set with train.py

  Prints out training loss, validation loss, and validation accuracy as the network trains: ```python train.py data_dir```

  Set directory to save checkpoints: python train.py data_dir --save_dir save_directory

  Choose architecture: python train.py data_dir --arch "alexnet"

  Set hyperparameters: python train.py data_dir --learning_rate 0.001 --epochs 3

  Use GPU for training: python train.py data_dir --gpu

- Predict flower name from an image with predict.py along with the probability of that name.

  Pass in a single image /path/to/image and return the flower name and class probability: python predict.py /path/to/image checkpoint

  Return top K most likely classes: python predict.py input checkpoint --top_k 3

  Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json

  Use GPU for inference: python predict.py input checkpoint --gpu
