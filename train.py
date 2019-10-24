import argparse
import torch
import numpy as np
import torch.nn.functional as func
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import AI as AI

parser = argparse.ArgumentParser(description='Neural network.')

parser.add_argument('--data_dir', action = 'store', default = '/home/workspace/ImageClassifier/flowers', dest = 'data',
                    help = 'Enter the path of training data.')

parser.add_argument('--arch', action='store',
                    dest = 'model', default = 'vgg16',
                    help= 'Enter pretrained model to use(The default is VGG-16).')

parser.add_argument('--save_dir', action = 'store',
                    dest = 'save', default = '/home/workspace/ImageClassifier/checkpoint.pth',
                    help = 'Enter location to save checkpoint.')

parser.add_argument('--learning_rate', action = 'store',
                    dest = 'learning', type=float, default = 0.001,
                    help = 'Enter learning rate for training the model, default is 0.001.')

parser.add_argument('--dropout', action = 'store',
                    dest='dropout', type=int, default = 0.05,
                    help = 'Enter dropout for training the model, default is 0.05.')

parser.add_argument('--hidden_units', action = 'store',
                    dest = 'hidden', type=int, default = 1000,
                    help = 'Enter number of hidden units in classifier.')

parser.add_argument('--epochs', action = 'store',
                    dest = 'epochs', type = int, default = 5,
                    help = 'Enter number of epochs to use during training.')

parser.add_argument('--gpu', action = "store_true", default = True, dest = 'gpu',
                    help = 'Turn GPU mode on or off.')

arg = parser.parse_args()

data_dir = arg.data
save_dir = arg.save
learning_rate = arg.learning
dropout = arg.dropout
hidden = arg.hidden
epochs = arg.epochs
gpu = arg.gpu
model = arg.model
                    
trainloader, testloader, validloader, train_data, test_data, valid_data = AI.load_data(data_dir)

model = getattr(models,model)(pretrained=True)
inputs = model.classifier[0].in_features
AI.build(model, inputs, hidden, dropout,120)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

model, optimizer = AI.train(model,trainloader, validloader, criterion, optimizer, gpu,epochs)
AI.test(model, testloader, gpu)
AI.save_model(model, train_data, optimizer, save_dir, epochs)