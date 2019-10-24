import argparse
import torch
import numpy as np
import torch.nn.functional as func
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import AI as AI
import json


parser = argparse.ArgumentParser(description='Predict image.')

parser.add_argument('--image_path', action='store',
                    default = '/home/workspace/ImageClassifier/flowers/test/100/image_07896',
                    help='Enter path to image.')

parser.add_argument('--save_dir', action='store',
                    dest='save_directory', default = '/home/workspace/ImageClassifier/checkpoint.pth',
                    help='Enter location to save checkpoint in.')

parser.add_argument('--arch', action='store',
                    dest='model', default='vgg16',
                    help='Enter pretrained model to use, default is VGG-16.')

parser.add_argument('--top_k', action='store',
                    dest='topk', type=int, default = 3,
                    help='Enter number of top most likely classes to view, default is 3.')

parser.add_argument('--cat_to_name', action='store',
                    dest='cat_name_dir', default = '/home/workspace/ImageClassifier/cat_to_name.json',
                    help='Enter path to image.')

parser.add_argument('--gpu', action="store_true", default=True, dest = 'gpu'
                     ,help= 'Turn GPU mode on or off, default is off.')

arg = parser.parse_args()

save_dir = arg.save_directory
image = arg.image_path
top_k = arg.topk
gpu_mode = arg.gpu
cat_names = arg.cat_name_dir

with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

pre_tr_model = arg.model
model = getattr(models,pre_tr_model)(pretrained=True)
loaded_model = AI.load_checkpoint(model, save_dir, gpu_mode)
processed_image = AI.processImage(image)

if gpu_mode == True:
    processed_image = processed_image.to('cuda')

probs, classes = AI.predict_Image(processed_image, loaded_model, top_k, gpu_mode)
print(probs)
print(classes) 

names = []
for i in classes:
    names += [cat_to_name[str(i)]]
    

print(f"Prediction: '{names[0]}'\nprobability: {round(probs[0]*100,4)}% ")
