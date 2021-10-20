import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
#from resnet18 import *
#from resnet18IB import *
from dataset import *
#from densenet import *
#import matplotlib.pyplot as plt
import time, os, copy, numpy as np
#from livelossplot import PlotLosses
from train_model import train_model
from PIL import Image
import argparse
#from prettytable import PrettyTable
#import shutil
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

parser = argparse.ArgumentParser()
parser.add_argument("--origmodel", type=str, default="", help="Path to original model")
parser.add_argument("--origtrain",type=str,default="",help="Path to trained orginal model to retrain or check acc")
parser.add_argument("--val",action='store_true', default=False,help="to only calidate")
parser.add_argument("--finetune_model", type=str, default="", help="Path to finetune_model model")
parser.add_argument("--resume", type=str, default="", help="Path to resume model")
parser.add_argument('--kml', type=int, nargs='+',default=[1/32], help='Variance for initializing IB parameters')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--ib_lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='learning rate')

opt = parser.parse_args()
print(opt)


data_transforms = {
    'train': transforms.Compose([
            #transforms.RandomCrop(64, padding=4),
            transforms.Resize(224, Image.BICUBIC),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ]),
    'val': transforms.Compose([
        transforms.Resize(224, Image.BICUBIC),
        #transforms.transforms.Resize(32),
        transforms.ToTensor(),
        #transforms.transforms.Resize(32),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ]),
}

data_dir = 'tiny-imagenet-200'
val_dataset = Dataset(os.path.join(data_dir, 'val/images'),os.path.join(data_dir, 'val','wnids.txt'),
                        os.path.join(data_dir, 'val','val_annotations.txt'),dtransform= data_transforms['val'], training=False)
image_datasets = Dataset(os.path.join(data_dir, 'train'),os.path.join(data_dir, 'val','wnids.txt'), os.path.join(data_dir, 'val',
                          'val_annotations.txt'),dtransform= data_transforms['val'], training= True)

#image_datasets = {'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])}
dataloaders = {'train': torch.utils.data.DataLoader(image_datasets, batch_size=100, shuffle=True, num_workers=64),
               'val': torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=64)}

dataset_sizes = {'train': len(image_datasets), 'val': len(val_dataset)}
print("\n dataset_size", dataset_sizes)

if not os.path.isdir('ResTinyimagenet_model'):
  os.makedirs('ResTinyimagenet_model')
"""if not os.path.isdir('DenTinyimagenet_model'):
  os.makedirs('DenTinyimagenet_model')"""
#model_ft = resnet18(pretrained=True)
model_ft = models.resnet18(pretrained=True)
#Finetune Final few layers to adjust for tiny imagenet input
model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 200)
if opt.origtrain !="":
  state_d= torch.load(opt.origtrain)
  #print("\n keys--> ",state_d.keys())
  model_ft.load_state_dict(torch.load(opt.origtrain))
  #print("\n epoch=", torch.load(opt.origtrain)["epoch"])

#model_ft.maxpool= nn.Sequential()
#model_ft.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,  bias=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)params.json
#state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
#model_ft.load_state_dict(state_dict)
print('    Total params: %.2fM' % (sum(p.numel() for p in model_ft.parameters())/1000000))
  
#count_parameters(model_ft)
#Multi GPU

#model_ft = torch.nn.DataParallel(model_ft, device_ids=[0, 1])

#Loss Function
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#Train
model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=200, val=opt.val)

#torch.save(model.state_dict(), f"'ResTinyimagenet_model'/{model.__class__.__name__}_acc{best_acc}.pth")
