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
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnext50_32x4d,ResNeXt50_32X4D_Weights
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights
from env import Env


class BirdModel(nn.Module):

  @classmethod
  def modelFromName(cls,name,db):
     if name.startswith("RN50v1"):
        return RN50_V1(name,db)
     if name.startswith("RX50v1"):
        return RX50_V1(name,db)
     if name.startswith("RX50v2"):
        return RX50_V2(name,db)
     if name.startswith("RN50v2"):
        return RN50_V2(name,db)
     if name.startswith("RN101v2"):
        return RN101_V2(name,db)  


  def __init__(self,modname,db):
    super().__init__()
    self.modname = modname
    self.db = db
    self.numCats = db.numClasses()
    self.modeldir = Env.TheEnv.modeldir
    self.fdir = f"{self.modeldir}/{db.sname}"
    self.fname = f"{self.fdir}/{self.modname}"
    self.buildModel()
    self.loaded = False
    self.loadModelState()

  def buildModel(self):
     raise NotImplemented()
  
  def getParamList(self):
      return None

  ##
  # These write and load state dict, need to create model first
  # ported from dlg project
  def writeModelState(self):
      if not os.path.exists(self.fdir):
        os.makedirs(self.fdir)
      ms = self.to(torch.device('cpu'))
      fname = f"{self.fname}.pt" 
      print(f"writing to {fname}.")
      torch.save(ms.state_dict(), fname)

  def writeModelStateToDat(self):
      if not os.path.exists(self.fdir):
        os.makedirs(self.fdir)
      ms = self.to(torch.device('cpu'))
      fname = f"{self.fname}.dat" 
      print(f"writing to {fname}.")
      with open(fname,"wb") as f:
        esd.save_state_dict(ms.state_dict(), f)

  def loadModelState(self):
      fname = f"{self.fname}.pt" # always load pt format
      if os.path.exists(fname):
        self.load_state_dict(torch.load(fname))
        self.loaded = True

  def writeModel(self,modeldir,modelname):
        ms = self.to(torch.device('cpu'))
        ms = torch.jit.script(ms)
        ms.save(modeldir + '/' + modelname)

  @staticmethod
  def loadModel(modeldir,modelname):
      model = torch.jit.load(modeldir + '/' + modelname)
      return model
  
class RN50_V1(BirdModel):
    def __init__(self,name,db,l2size=256):
        self.l2size = l2size
        super().__init__(name,db)

    def buildModel(self):  # need this so supar cann call to init
        
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

                  nn.Linear(512,self.l2size),
                  nn.ReLU(),
                  nn.Dropout(0.5),
    #              nn.BatchNorm1d(),
                  nn.Linear(self.l2size, self.numCats),
                )
        
    def forward(self,x):
        x = self.rn50_model(x)
        x = self.bird_model(x)
        return x

    
## resNext -- train all layers
# don't freeze trained layers
class RX50_V2(BirdModel):
    def __init__(self,name,db,l2size=256):
        self.l2size = l2size
        super().__init__(name,db)

    def buildModel(self):  # need this so supar cann call to init
        
        self.rn50_model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
       
        num_ftrs = self.rn50_model.fc.in_features
        self.rn50_model.fc = nn.Identity()
        # Parameters of newly constructed modules have requires_grad=True by default
        self.bird_model = nn.Sequential(
                  nn.Linear(num_ftrs,self.numCats) # simple, just one layer (inspired by https://github.com/ecm200/caltech_birds)
                 
                )
        
    def forward(self,x):
        x = self.rn50_model(x)
        x = self.bird_model(x)
        return x

class RX101_V2(BirdModel):
    def __init__(self,name,db,l2size=256):
        self.l2size = l2size
        super().__init__(name,db)

    def buildModel(self):  # need this so supar cann call to init
        
        self.rn50_model = resnext101_64x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
       
        num_ftrs = self.rn50_model.fc.in_features
        self.rn50_model.fc = nn.Identity()
        # Parameters of newly constructed modules have requires_grad=True by default
        self.bird_model = nn.Sequential(
                  nn.Linear(num_ftrs,self.numCats) # simple, 
                 
                )
        
    def forward(self,x):
        x = self.rn50_model(x)
        x = self.bird_model(x)
        return x   
    
# don't freeze trained layers
class RN50_V2(BirdModel):
    def __init__(self,name,db,l2size=256):
        self.l2size = l2size
        super().__init__(name,db)

    def buildModel(self):  # need this so supar cann call to init
        
        self.rn50_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
       
        num_ftrs = self.rn50_model.fc.in_features
        self.rn50_model.fc = nn.Identity()
        # Parameters of newly constructed modules have requires_grad=True by default
        self.bird_model = nn.Sequential(
                  nn.Linear(num_ftrs,self.numCats) # simple, just one layer (inspired by https://github.com/ecm200/caltech_birds)
                 
                )
        
    def forward(self,x):
        x = self.rn50_model(x)
        x = self.bird_model(x)
        return x

class RN101_V2(BirdModel):
    def __init__(self,name,db,l2size=256):
        self.l2size = l2size
        super().__init__(name,db)

    def buildModel(self):  # need this so supar cann call to init
        
        self.rn50_model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        num_ftrs = self.rn50_model.fc.in_features
        self.rn50_model.fc = nn.Identity()
        # Parameters of newly constructed modules have requires_grad=True by default
        self.bird_model = nn.Sequential(
                  nn.Linear(num_ftrs,self.numCats) # simple, just one layer (inspired by https://github.com/ecm200/caltech_birds)
                 
                )
        
    def forward(self,x):
        x = self.rn50_model(x)
        x = self.bird_model(x)
        return x