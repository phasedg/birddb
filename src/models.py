import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms.v2 as transforms
import sys
from PIL import Image
import os
import torch.nn as nn
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights



class RN50_Bird_V1(nn.Module):
    def __init__(self,name,numCats,l2size=256):
        super().__init__()
        self.name = name
        self.rn50_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for param in self.rn50_model.parameters():
            param.requires_grad = False
        for param in self.rn50_model.layer4.parameters():
            param.requires_grad = True
        for param in self.rn50_model.avgpool.parameters():
            param.requires_grad = True
        num_ftrs = self.rn50_model.fc.in_features
        self.rn50_model.fc = nn.Identity()
        # Parameters of newly constructed modules have requires_grad=True by default
        self.bird_model = nn.Sequential(
                  nn.Linear(num_ftrs,512),
                  nn.ReLU(),
                  nn.Dropout(0.5),

                  nn.Linear(512,l2size),
                  nn.ReLU(),
                  nn.Dropout(0.5),
    #              nn.BatchNorm1d(),
                  nn.Linear(l2size, numCats),
                )
        
    def forward(self,x):
        x = self.rn50_model(x)
        x = self.bird_model(x)
        return x

    def writeModel(self,modeldir,modelname):
        ms = self.to(torch.device('cpu'))
        ms = torch.jit.script(ms)
        ms.save(modeldir + '/' + modelname)

    @staticmethod
    def loadModel(modeldir,modelname):
        model = torch.jit.load(modeldir + '/' + modelname)
        return model


        
