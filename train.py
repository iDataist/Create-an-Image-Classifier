import numpy as np
import torch 
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse

"""
Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu
"""
    
def load_train_data(data_dir):
    transform = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],   
                                                          [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    data = datasets.ImageFolder(data_dir, transform =  transform)
    
    # Using the image datasets and the trainforms, define the dataloaders
    dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    return data, dataloader 

def load_validation_or_test_data(data_dir):    
    transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    data = datasets.ImageFolder(data_dir, transform = transform)

    # Using the image datasets and the trainforms, define the dataloaders
    dataloader = torch.utils.data.DataLoader(data, batch_size=64)
    return data, dataloader

def build_model(architecture, input_units, hidden_units, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = getattr(models, architecture)(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    if model.classifier: 
        model.classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(input_units, hidden_units[0])),
                                  ('relu', nn.ReLU()),
                                  ('dropout1', nn.Dropout(0.3)), 
                                  ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
                                  ('relu', nn.ReLU()),
                                  ('dropout2', nn.Dropout(0.3)), 
                                  ('fc3', nn.Linear(hidden_units[1], 102)),    
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))

        criterion = nn.NLLLoss()

        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

        model.to(device)    
        return model, optimizer, criterion
    else:
        print('The archetecture does not have the classifier attribute')
        
def train_and_validate(model, optimizer, criterion, trainloader, validloader, epochs): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = 0
    running_loss = 0
    print_every = 32
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    return model

def test(model, criterion, testloader): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():    
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")

def save_checkpoint(model, train_data, save_directory):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'classifier': model.classifier, 
                  'class_to_idx': model.class_to_idx, 
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, save_directory)

def main(data_dir, gpu, architecture, input_units, hidden_units, learning_rate, epochs, save_directory):     
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'   
    
    train_data, trainloader = load_train_data(train_dir)
    valid_data, validloader = load_validation_or_test_data(valid_dir)
    test_data, testloader = load_validation_or_test_data(test_dir)
    
    device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")
    
    model, optimizer, criterion = build_model(architecture, input_units, hidden_units, learning_rate)
    model = train_and_validate(model, optimizer, criterion, trainloader, validloader, epochs)
    test(model, criterion, testloader)
    
    save_checkpoint(model, train_data, save_directory)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Train a image classifier using transfer learning')

    parser.add_argument('--data_dir', default = 'ImageClassifier/flowers', action='store_true')
    parser.add_argument('--gpu', default = True, action='store_true')
    parser.add_argument('--architecture', default = 'alexnet')
    parser.add_argument('--input_units', default = 9216, type = int)
    parser.add_argument('--hidden_units', nargs = '+', default = [2014, 512], type = int)
    parser.add_argument('--learning_rate', default = 0.001, type = float)
    parser.add_argument('--epochs', default = 1, type = int)
    parser.add_argument('--save_directory', default = 'ImageClassifier/checkpoint.pth', action='store_true')
    
    args = parser.parse_args()
    main(args.data_dir, args.gpu, args.architecture, args.input_units, args.hidden_units, args.learning_rate, args.epochs, args.save_directory)
