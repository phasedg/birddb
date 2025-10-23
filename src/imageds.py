import torch
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torchvision.io import read_image,ImageReadMode
from pathlib import Path 
import random
import os
import shutil
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as transforms
import torchvision.io as tu
import torchvision
from PIL import Image
from collections import namedtuple
from env import Env
from birddb import BirdDB

BdImage = namedtuple("BdImage",["Id","TrainTest","ClassId","ImFile","ClassIdx"])
BdClass = namedtuple("BdClass",["ClassId","ClassName","Index","Parent"])




RESNET_IMSIZE = 224

'''
Base class for image data set. Loaded data into mem. If DB too large need new approach
'''
class ImageDataset(Dataset):
    
    if torchvision.__version__[0:4] == '0.15':
        RESIZE_TRANS = transforms.Compose([
            transforms.Resize([RESNET_IMSIZE,RESNET_IMSIZE],antialias=True),
            transforms.ConvertImageDtype(torch.float32),
        ])
    else:
        RESIZE_TRANS = transforms.Compose([
            transforms.Resize([RESNET_IMSIZE,RESNET_IMSIZE],antialias=True),
            transforms.ToDtype(torch.float32,scale=True),
          
        ])

    def __init__(self, db,  transform=None,todev = True):
        
        self.transform = transform
        self.db = db
        self.todev = todev
       
        
        self.imageTens = []
        self.labelTens = []
        
        self.name = None
        for i in range(0,db.numImages()):
          img_path, label = self.db.imageFileAndLabel(i) # returns class index as label 
          ## this reads into tensor
          image = torchvision.io.decode_image(img_path,ImageReadMode.RGB) #some CBU in BW
          if image.shape[0] < 3:
            print(image.shape,img_path)
          image = ImageDataset.RESIZE_TRANS(image)  # resize and scale
          label = torch.tensor(label)
          if todev:
              image = image.to(Env.TheEnv.device)
              label = label.to(Env.TheEnv.device)
          self.imageTens.append(image)
          self.labelTens.append(label)
          if i%100 == 0:
              print('.',end='')
        print("Loaded")

    def numClasses(self):
        return self.db.numClasses()

    def __len__(self):
        return self.db.numImages();

    def __getitem__(self, idx):

        image = self.imageTens[idx]
        label = self.labelTens[idx]
        
        if self.transform:
          image = self.transform(image)  # possible run transform pipeline here
        
        return image, label  # returns tensor and int

    def getClassName(self,idx):
        return self.db.className(idx)

        
    def makeDataloader(self,batch_size,shuffle=True):         
        return DataLoader(self,batch_size,shuffle)

        
    def showImages(self,w=4,h=5,figsize=(10,10)):
        fig = plt.figure(figsize=figsize)
        if self.name is not None:
            fig.suptitle(self.name)
        for i in range(0,w*h):
            sample_idx = random.randint(0,len(self)-1)
            img, label = self[sample_idx]   
            #print(img.shape)
            ax = fig.add_subplot(w,h,i+1)
            ax.imshow(img.permute(1,2,0))
            ax.set_title(f'{sample_idx} : {label} \n{self.getClassName(label)}', fontsize=8)
        plt.show()

    def showLoaderImages(self,batch_size=32,shuffle=False,w=4,h=5,figsize=(10,10)):
        dl = self.makeDataloader(batch_size,shuffle)
        fig = plt.figure(figsize=figsize)
        if self.name is not None:
            fig.suptitle('Loader: '+ self.name)
        count = 0
        while count < w*h:
            batch_imgs, batch_labels = next(iter(dl))            
            for i in range(0,batch_imgs.shape[0]):
                img, label = batch_imgs[i].squeeze(), batch_labels[i]
                #print(img.shape)
                ax = fig.add_subplot(w,h,count+1)
                ax.imshow(img.permute(1,2,0))
                ax.set_title(self.getClassName(label), fontsize=8)
                count = count+1
                if count >= w*h:
                    break
        plt.show()


'''
Iter class, this is to try to reduce CPU GPU transfers. Not really sufficient for large dataset and doesn't shuffle.
'''
class IterImageset(IterableDataset):
    
    RESIZE_TRANS = transforms.Compose([
        transforms.Resize([RESNET_IMSIZE,RESNET_IMSIZE],antialias=True),
        transforms.ToDtype(torch.float32,scale=True),
      
    ])

    # read DB into mem, batch and move to device
    def __init__(self, db, batchsize , todev = False):
        
        
        self.db = db
        self.batchsize = batchsize
        self.todev = todev
      
        self.imageBatches = []
        self.labelBatches = []
        
        self.name = None

        for bcount in range(0,self.count()//self.batchsize): # Ignores last fragment
          images = []
          labels = []
          for i in range(0,self.batchsize):
            idx = bcount+i
            img_path, label = self.db.imageFileAndLabel(idx)
            image = torchvision.io.decode_image(img_path,ImageReadMode.RGB) #some CBU in BW
            image = ImageDataset.RESIZE_TRANS(image)  # resize and scale
            images.append(image)
            labels.append(torch.tensor(label))
          ibatch = torch.stack(images)
          lbatch = torch.stack(labels)
          if todev:
              ibatch = ibatch.to(Env.TheEnv.device)
              lbatch = lbatch.to(Env.TheEnv.device)
          self.imageBatches.append(ibatch)
          self.labelBatches.append(lbatch)
          print(".",end='')
        print("loaded")
        self.dataset = db

    def __iter__(self):
      self.idx = 0
      return self
    
    def __next__(self):
        if self.idx >= len(self.imageBatches):
            raise StopIteration
        else:
            val = (self.imageBatches[self.idx],self.labelBatches[self.idx])
            self.idx += 1
            return val
        


        
            

    def numClasses(self):
        return self.db.numClasses()

    def count(self):
        return self.db.numImages();
    
    def __len__(self):
        return self.db.numImages();
        
        if False and self.transform:
          image = self.transform(image)  # possible run transform pipeline here
        if image.shape[0] < 3:
            print(image.shape,img_path)
        return image, label  # returns tensor and int

    def getClassName(self,idx):
        return self.db.className(idx)

        
    def makeDataloader(self,batch_size=1):     # batch_size dummy for compat    
        return DataLoader(self,batch_size=None)

        
    def showImages(self,w=4,h=5,figsize=(10,10)):
        fig = plt.figure(figsize=figsize)
        if self.name is not None:
            fig.suptitle(self.name)
        for i in range(0,w*h):
            sample_idx = random.randint(0,len(self)-1)
            img, label = self[sample_idx]   
            #print(img.shape)
            ax = fig.add_subplot(w,h,i+1)
            ax.imshow(img.permute(1,2,0))
            ax.set_title(f'{sample_idx} : {label} \n{self.getClassName(label)}', fontsize=8)
        plt.show()

    def showLoaderImages(self,batch_size=32,shuffle=False,w=4,h=5,figsize=(10,10)):
        dl = self.makeDataloader(batch_size,shuffle)
        fig = plt.figure(figsize=figsize)
        if self.name is not None:
            fig.suptitle('Loader: '+ self.name)
        count = 0
        while count < w*h:
            batch_imgs, batch_labels = next(iter(dl))            
            for i in range(0,batch_imgs.shape[0]):
                img, label = batch_imgs[i].squeeze(), batch_labels[i]
                #print(img.shape)
                ax = fig.add_subplot(w,h,count+1)
                ax.imshow(img.permute(1,2,0))
                ax.set_title(self.getClassName(label), fontsize=8)
                count = count+1
                if count >= w*h:
                    break
        plt.show()  


if __name__ == "__main__":
    datadir = '/data1/datasets/birds'
    sname = 'nab_wood_sm'
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Env.initEnv("/home/dg/proj/birddb",datadir,device)
    ##driver = Driver(expbase,datadir,dbname,device)
    db = BirdDB.DBFromName(sname)
    ds = IterImageset(db,batchsize=8)
    for im,lab in ds:
        print(im,lab)
        break
    for im,lab in ds.makeDataloader():
        print(im,lab)
        break
    
 
            

