from ntpath import isfile
import time
import os
import tempfile
import torch
import models
from pathlib import Path
import birddb
import models
from trainer import Trainer
from torchvision.transforms import v2 as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import glob
from env import Env
from birddb import BirdDB
from imageds import ImageDataset
from imageds import IterImageset
from models import BirdModel
from testRun import TestRun
from collections import namedtuple

ExptArgs = namedtuple('ExptArgs',['ExptName','Epochs','BatchSize','LR','Decay'])
'''
  New Expt series. Old expt.py kept for documentation
'''
class Expt2:
    
    @classmethod
    def ExptFromName(cls,exptname,db):
      fields = exptname.split('_')   #<modclass>_<expt>_<trainargs>
      modname = fields[0]
      ename = fields[1]
      epochs = 20
      batch_size = 32
      lr = 0.01
      decay = (4,0.8)
      for f in fields[2:]:
          if f[0] == 'e':
            epochs = int(f[1:])
          if f[0] == 'b':
            batch_size = int(f[1:])
          if f[0] == 'l':
            lr = int(f[1:]) * 0.001 # l05 => 0.005
          if f[0] == 'd':
            decay = (int(f[1]),0.1 * int(f[2]))  # steps, factor (8 => 0.8)
          if f == 'nv':
            useval = False  # don't validate in training (faster)
      nargs = ExptArgs(exptname,epochs,batch_size,lr,decay)
      if ename == "u1":
          return Expt2_U1(db,nargs)
      if ename == "u2":
          return Expt2_U2(db,nargs)
      

      raise Exception(f"No expt for {ename}, {exptname}")

    def __init__(self,db,nargs):
        self.datadir = Env.TheEnv.datadir
        self.db = db
        self.expdir = f"{Env.TheEnv.expdir}/{db.sname}"
        if not os.path.exists(self.expdir):
            os.makedirs(self.expdir)
        self.rundir = f"{self.expdir}/{nargs.ExptName}"
        self.runStatFile = f"{self.rundir}/trainStats.csv"

        self.device = Env.TheEnv.device
        self.chkpath = f"{self.rundir}/check.pt"
        self.model = None
        self.trainTrans = None
        self.nargs = nargs
        self.epochs = nargs.Epochs
        self.batch_size = nargs.BatchSize
        self.todev = True
        self.useval = True
        self.exptname = nargs.ExptName
        
        self.trainer = None


    def callback(self,epoch):
        if (epoch+1) % 5 == 0:
            if not os.path.isdir(os.path.dirname(self.chkpath)):  #should be self.runDir but better to be sure
                os.makedirs(os.path.dirname(self.chkpath))
            self.trainer.checkpoint(self.chkpath)

    def hasCheckpoint(self):
        print(self.chkpath)
        return os.path.isfile(self.chkpath)
    
    def trainModel(self,mod,write=True,dotfreq=10):
        self.trainer = self.buildTrainer(mod)
        self.trainer.dotfreq=dotfreq
        if self.hasCheckpoint():
            self.trainer.restore(self.chkpath)
        stats = self.trainer.train(self.epochs,self.callback)
        # write model
        mod.writeModelState()
        # clean up checkpt file
        if os.path.isfile(self.chkpath):
            os.remove(self.chkpath)
        #self.trainer.model.writeModel(self.expdir,'model.pt')
        if not os.path.isdir(os.path.dirname(self.runStatFile)):  #should be self.runDir but better to be sure
            os.makedirs(os.path.dirname(self.runStatFile))
        with open(self.runStatFile,"w") as f:
            f.write(f"Loss,Acc,VLoss,Vacc,ETime,GTime\n")
            for x in stats:
                f.write(f"{x[0]:.3f},{x[1]:.3f},{x[2]:.3f},{x[3]:.3f},{x[4]:.3f},{x[5]:.3f}\n")
                
    
    def getTrainRun(self,tstdb):
        return TestRun(self.rundir,self.exptname,self.db,tstdb.sname)

    ## runs model on DB, writes file
    ##  imageID, <top5 class ids> <tops 5 probs>
    def runOnDB(self,db,batchsize=32):
        since = time.time()
        moddb = self.db
        tr = TestRun(self.rundir,self.exptname,moddb,db.sname)
        if len(tr) > 0:
            return tr
        best_acc = 0.0
        ds = ImageDataset(db,self.valTrans,True)
        self.model.to(self.device)
        # Each epoch has a training and validation phase
        self.model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        batch = []
        ids = []
        # Iterate over data. batch to improve GPU utilization
        for i in range(0,len(ds)):
            imId = db.imageId(i)
            input, _ = ds[i]
            batch.append(input)
            ids.append(imId)
            if len(batch) < batchsize and i != len(ds)-1:
              continue
            inputs = torch.stack(batch)
            inputs = inputs.to(self.device)
            if (i+1)%(self.batch_size*10) == 0:
                print('.',end='')
            
            with torch.no_grad():
                outputs = self.model(inputs) # tensor (batchsize,numclasses)
                outputs = nn.functional.softmax(outputs,dim=1) # probs for classes
                for j in range(0,outputs.shape[0]):
                  res = []
                  for r in range(0,outputs.shape[1]):
                    k = (moddb.classId(r),outputs[j][r].item())  # look up result in madel dict (classId,prob)
                    res.append(k)
                  res = sorted(res,key=lambda x: -x[1]) # sort by output desc
                  res = res[0:5]
                  tr.add(ids[j],res)
            batch = []
            ids = []
        tr.write()
        print()
        return tr

    def visualizeOnDataset(self,ds, nameMap = None,num_images=6):
        dl = ds.makeDataloader(batch_size=32, shuffle=True)
        self.model.to(self.device)
        self.model.eval()
        images_so_far = 0
        fig = plt.figure(figsize=(10,10))
        if nameMap is None:
            nameMap = ds.getClassName
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dl):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'label: {nameMap(labels[j])} \npredicted:{ nameMap(preds[j])}',fontsize=8)
                    ax.imshow(inputs.cpu().data[j].permute(1,2,0))
                    if images_so_far == num_images:
                        break
                if images_so_far == num_images:
                    break
        plt.show()


#refactored version of Expt_t10
class Expt2_U1(Expt2):

    
    def __init__(self,db,args):
        RESNET_IMSIZE = 224
        DEF_MEANS = [0.485, 0.456, 0.406]
        super().__init__(db,args)
        self.trainTrans = torch.nn.Sequential(
            
            # transforms.AutoAugment(),
            transforms.Resize([RESNET_IMSIZE,RESNET_IMSIZE],antialias=True),
            transforms.RandomHorizontalFlip(),

            transforms.RandomChoice([
            transforms.ColorJitter(brightness=.3,hue=0.1,contrast=0.2),
            transforms.RandomRotation(20),
            transforms.RandomAutocontrast(0.2)],
            [0.6,0.6,0.6]),
            
            transforms.ToDtype(torch.float32,scale=True),
            transforms.Normalize(mean=db.means, std=[0.229, 0.224, 0.225]),
            # maybe normalize
            #    transforms.CenterCrop(224),
        )
        self.valTrans = torch.nn.Sequential(
            transforms.Resize([RESNET_IMSIZE,RESNET_IMSIZE],antialias=True),
            transforms.ToDtype(torch.float32,scale=True),
            transforms.Normalize(mean=db.means, std=[0.229, 0.224, 0.225]),
            )
        self.criterion = nn.CrossEntropyLoss()
        

    def buildTrainer(self,model):
        traindb = self.db.getTrainDB()
        lr = self.nargs.LR
        decay = self.nargs.Decay
        valdl = None
        
        self.trainds = ImageDataset(traindb,transform=self.trainTrans,todev=self.todev,rescale=False)
        self.traindl = self.trainds.makeDataloader(batch_size=self.batch_size,shuffle=True)
        if self.useval:
            valdb = self.db.getValDB()
            valds = ImageDataset(valdb,transform=self.valTrans,todev=self.todev)
            valdl = valds.makeDataloader(batch_size=self.batch_size,shuffle=False)
        params = model.getParamList()
        if params is None:
            params = [{'params':model.bird_model.parameters()},
                      {'params':model.rn50_model.parameters()}]
            
        self.optimizer = optim.SGD(params,
                                   lr=lr, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        print(f"Sched: step {decay[0]}, gamma {decay[1]}")
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=decay[0], gamma= decay[1])
        self.trainer = Trainer(model, self.device, self.traindl, valdl, self.criterion, self.optimizer, self.scheduler, writer=None)
        return self.trainer
   

#
# uses AutoTranform
class Expt2_U2(Expt2_U1):

    
    def __init__(self,db,args):
        RESNET_IMSIZE = 224
        DEF_MEANS = [0.485, 0.456, 0.406]
        super().__init__(db,args)
        self.trainTrans = torch.nn.Sequential(
            
            # 
            transforms.Resize([RESNET_IMSIZE,RESNET_IMSIZE],antialias=True),

            transforms.AutoAugment(),

            transforms.ToDtype(torch.float32,scale=True),
            transforms.Normalize(mean=db.means, std=[0.229, 0.224, 0.225]),
            # maybe normalize
            #    transforms.CenterCrop(224),
        )
        self.valTrans = torch.nn.Sequential(
            transforms.Resize([RESNET_IMSIZE,RESNET_IMSIZE],antialias=True),
            transforms.ToDtype(torch.float32,scale=True),
            transforms.Normalize(mean=db.means, std=[0.229, 0.224, 0.225]),
            )
        self.criterion = nn.CrossEntropyLoss()
        

    def buildTrainer(self,model):
        traindb = self.db.getTrainDB()
        lr = self.nargs.LR
        decay = self.nargs.Decay
        valdl = None
        
        self.trainds = ImageDataset(traindb,transform=self.trainTrans,todev=self.todev,rescale=False)
        self.traindl = self.trainds.makeDataloader(batch_size=self.batch_size,shuffle=True)
        if self.useval:
            valdb = self.db.getValDB()
            valds = ImageDataset(valdb,transform=self.valTrans,todev=self.todev)
            valdl = valds.makeDataloader(batch_size=self.batch_size,shuffle=False)
        params = model.getParamList()
        if params is None:
            params = [{'params':model.bird_model.parameters()},
                      {'params':model.rn50_model.parameters()}]
            
        self.optimizer = optim.SGD(params,
                                   lr=lr, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        print(f"Sched: step {decay[0]}, gamma {decay[1]}")
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=decay[0], gamma= decay[1])
        self.trainer = Trainer(model, self.device, self.traindl, valdl, self.criterion, self.optimizer, self.scheduler, writer=None)
        return self.trainer
   
