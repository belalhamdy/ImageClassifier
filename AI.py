import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as Func
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image


def load_data (data_dir = 'flowers' ,batchsize=64):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]) 

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]) 

    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    validation_data = datasets.ImageFolder(data_dir + '/valid', transform=validation_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test' ,transform = test_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_data, batch_size =batchsize,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = batchsize, shuffle = True)
    return trainloader, testloader, validloader, train_data, test_data, validation_data

def build (model,lastIn,hidden,dropout,out):
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(lastIn, hidden)),('relu', nn.ReLU()),('dropout1', nn.Dropout(dropout)),
                              ('fc2', nn.Linear(hidden, out)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    return model

def validation(model, validloader, criterion,gpu =  True):
    vloss = 0
    accuracy = 0
    if (gpu == True):
        model.to('cuda')
    for ii, (images, labels) in enumerate(validloader):
        if (gpu == True):
            images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        vloss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return vloss, accuracy

def train(model,trainloader, validloader, criterion, optimizer,gpu =  True,epochs=3,print_every=10):
    steps = 0
    if (gpu == True):
        model.to('cuda')
    
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            if (gpu == True):
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    vloss, accuracy = validation(model, validloader, criterion)
                print(f"Epoch: {e+1},{ii+1} \
                TrainingLoss: {round(running_loss/print_every,5)} \
                ValidLoss: {round(vloss/len(validloader),5)} \
                ValidAccuracy: {round(float(accuracy/len(validloader)),5)}")
                running_loss = 0
                model.train()
    
    return model, optimizer

def test(model, testloader , gpu =  True):
    correct = 0
    total = 0
    if (gpu == True):
        model.to('cuda')
    with torch.no_grad():
        for ii, (images, labels) in enumerate(testloader):
            if (gpu == True):
                images, labels = images.to('cuda'), labels.to('cuda')           
            outputs = model(images)
            total += labels.size(0)
            temp, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            ans = correct/total
    print(f"Accuracy =  {ans}")
    
def processImage(image):
    pil = Image.open(f'{image}' + '.jpg')
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    nparray = np.array(transform(pil))
    imgtensor = torch.from_numpy(nparray).type(torch.FloatTensor)
    return imgtensor.unsqueeze_(0)
 
def save_model(model, train_data, optimizer, save_dir, epochs):
    checkpoint = {'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': train_data.class_to_idx,
                  'opt_state_dict': optimizer.state_dict,
                  'epochs': epochs}
    return torch.save(checkpoint, save_dir)

def load_checkpoint(model, save_dir,gpu = True):
    if gpu == True:
        checkpoint = torch.load(save_dir)
    else:
        checkpoint = torch.load(save_dir, map_location=lambda storage, loc: storage)
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def predict_Image(image, model, topk,gpu):
    model.eval()
    if (gpu == True):
        model.to('cuda') 
    with torch.no_grad():
        output = model.forward(image)

    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]
    
    Ptop_list = np.array(probs_top)[0]
    Itop_list = np.array(index_top[0])
    
    class_to_idx = model.class_to_idx
    indx_to_class = {a: b for b, a in class_to_idx.items()}
    classes_top_list = []
    for index in Itop_list:
        classes_top_list += [indx_to_class[index]]       
    return Ptop_list, Itop_list