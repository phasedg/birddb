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
'''
  Expt subclasses encapsule a model structure, training regime
  Instances add training dataset, and paths
'''
class Expt:
    
    @classmethod
    def ExptFromName(cls,exptname,db):
      fields = exptname.split('_')   #<modclass>_<expt>_<trainargs>
      modname = fields[0]
      ename = fields[1]
      epochs = 4
      batch_size = 32
      todev = True
      useval = True
      for f in fields[2:]:
          if f[0] == 'e':
            epochs = int(f[1:])
          if f[0] == 'b':
            batch_size = int(f[1:])
          if f == 'dc':
            todev = False  # don't move data to device on load
          if f == 'nv':
            useval = False  # don't move data to device on load
      args = (epochs,batch_size,todev,useval)
      if ename == "t1":
          return Expt_T1(db,args)
      if ename == "t2":
          return Expt_T2(db,args)
      if ename == "t1i":
          return Expt_T1i(db,args)
      raise Exception(f"No expt for {ename}, {exptname}")

    def __init__(self,db,args,trainTrans=None):
        self.datadir = Env.TheEnv.datadir
        self.db = db
        self.expdir = f"{Env.TheEnv.expdir}/{db.sname}"
        if not os.path.exists(self.expdir):
            os.makedirs(self.expdir)

        self.device = Env.TheEnv.device
        self.chkpath = self.expdir + '/check.pt'
        self.model = None
        self.trainTrans = trainTrans
        self.args = args
        self.epochs = args[0]
        self.batch_size = args[1]
        self.todev = args[2]
        
        self.trainer = None

    def dirDict(self):
        return {
            'datadir': self.datadir,
            'dbname': self.dbname,
            'expdir': self.expdir,
            'chkpath': self.chkpath,
            'dbdir': self.dbdir
            }

    def callback(self,epoch):
        if (epoch+1) % 5 == 0:
            self.trainer.checkpoint(self.chkpath)

    def hasCheckpoint(self):
        print(self.chkpath)
        return os.path.isfile(self.chkpath)
    
    def trainModel(self,mod,write=True,dotfreq=10):
        self.trainer = self.buildTrainer(mod)
        self.trainer.dotfreq=dotfreq
        if self.hasCheckpoint():
            self.trainer.restore(self.chkpath)
        stats = self.trainer.train(self.epochs,None)
        #self.trainer.model.writeModel(self.expdir,'model.pt')
        fname = f"{self.expdir}/{mod.modname}"
        if not os.path.isdir(fname):
            os.makedirs(fname)
        fname = f"{fname}/trainStats.csv"
        with open(fname,"w") as f:
            f.write(f"Loss,Acc,VLoss,Vacc,ETime,GTime\n")
            for x in stats:
                f.write(f"{x[0]:.3f},{x[1]:.3f},{x[2]:.3f},{x[3]:.3f},{x[4]:.3f},{x[5]:.3f}\n")
                
                


    def loadModel(self):
        self.model = models.RN50_Bird_V1.loadModel(self.expdir,'model.pt')
        return self.model

    def testOnDataset(self,ds):
        since = time.time()
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        dl = ds.makeDataloader(batch_size=32, shuffle=False)
        self. model.to(self.device)
        # Each epoch has a training and validation phase
        self.model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        # Iterate over data.
        for inputs, labels in dl:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                # statistics
            running_loss += loss.item() * inputs.size(0)
                # print(outputs.shape,preds.shape,labels.data.shape)
            running_corrects += torch.sum(preds == labels)
        epoch_loss = running_loss / len(ds)
        epoch_acc = running_corrects.double() / len(ds)
        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        time_elapsed = time.time() - since
        print(f'Testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        return epoch_loss, epoch_acc

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

    def crossTest(self,dbs):
        stats = {}
        statfile = self.expdir + '/stats.pkl'
        if os.path.exists(statfile):
            with open(statfile,'rb') as f:
                stats = pickle.load(f)
        expbase = os.path.dirname(self.expdir)  #strip train db level 
        for db in dbs:
            if db not in stats:
                stats[db] = {}
            for test in ['train', 'test']:
                if test not in stats[db]:
                    print(f'{test} not in stats[{db}]...computing')
                    stats[db][test] = {}
                    dbdir = Path(self.datadir / db)
                    if db[0] == self.dbname[0]: # quicl test
                        testds = birddb.TestImageDataset(dbdir,self.trainds,trainset=test)
                    else:
                        testds = birddb.CrossImageDataset(dbdir,self.trainds,trainset=test)
                    testds.printStats()
                    print(f'db: {db}, test: {test}')
                    #myexpt.visualizeOnDataset(testds)
                    loss, acc = self.testOnDataset(testds)
                    stats[db][test]= acc.item()
                    with open(statfile,'wb') as f:
                        pickle.dump(stats,f)

        rows = []
        for fname in glob.glob(expbase + '/*/*.pkl'):
            dbname = fname.split('/')[-2]
            print(dbname)
            with open(fname,'rb') as f:
                stats = pickle.load(f)
                print(stats)
                for x in stats:
                    row = [dbname,x,stats[x]['train'],stats[x]['test']]
                    rows.append(row)
        df = pd.DataFrame(rows,columns=['traindb','testdb','trainacc','testacc'])
        df.to_csv(expbase + '/stats.csv',index=False)
        return df

    def combineStats(self):
        rows = []
        expbase = os.path.dirname(self.expdir)  #strip train db level 
        for fname in glob.glob(expbase + '/*/*.pkl'):
            dbname = fname.split('/')[-2]
            print(dbname)
            with open(fname,'rb') as f:
                stats = pickle.load(f)
                print(stats)
                for x in stats:
                    row = [dbname,x,stats[x]['train'],stats[x]['test']]
                    rows.append(row)
        df = pd.DataFrame(rows,columns=['traindb','testdb','trainacc','testacc'])
        df.to_csv(expbase + '/stats.csv',index=False)
        return df


# similar ot trainproc in dlg
class Expt_T1(Expt):
    def __init__(self,db,args):
        trainTrans = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.AutoAugment(),
            #transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=.3),
            # transforms.RandomRotation(20),
            #    transforms.CenterCrop(224),
        ])
        super().__init__(db,args,trainTrans)
        self.criterion = nn.CrossEntropyLoss()



    def buildTrainer(self,model):
        traindb = self.db.getTrainDB()
        useval = self.args[3]
        valdl = None
        
        self.trainds = ImageDataset(traindb,transform=self.trainTrans,todev=self.todev)
        self.traindl = self.trainds.makeDataloader(batch_size=self.batch_size,shuffle=True)
        if useval:
            valdb = self.db.getValDB()
            valds = ImageDataset(valdb,transform=None,todev=self.todev)
            valdl = valds.makeDataloader(batch_size=self.batch_size,shuffle=False)
        self.optimizer = optim.SGD([{'params':model.bird_model.parameters()},
                            {'params':model.rn50_model.avgpool.parameters()},
                            {'params':model.rn50_model.layer4.parameters()}],
                                   lr=0.01, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma= 0.9)
        self.scheduler = None
        self.trainer = Trainer(model, self.device, self.traindl, valdl, self.criterion, self.optimizer, self.scheduler, writer=None)
        return self.trainer
    
class Expt_T2(Expt):
    def __init__(self,db,args):
        trainTrans = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.AutoAugment(),
            #transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=.3),
            # transforms.RandomRotation(20),
            #    transforms.CenterCrop(224),
        ])
        self.valTrans = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #transforms.AutoAugment(),
            #transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=.3),
            # transforms.RandomRotation(20),
            #    transforms.CenterCrop(224),
        ])
        super().__init__(db,args,trainTrans)
        self.criterion = nn.CrossEntropyLoss()



    def buildTrainer(self,model):
        traindb = self.db.getTrainDB()
        useval = self.args[3]
        valdl = None
        
        self.trainds = ImageDataset(traindb,transform=self.trainTrans,todev=self.todev)
        self.traindl = self.trainds.makeDataloader(batch_size=self.batch_size,shuffle=True)
        if useval:
            valdb = self.db.getValDB()
            valds = ImageDataset(valdb,transform=self.valTrans,todev=self.todev)
            valdl = valds.makeDataloader(batch_size=self.batch_size,shuffle=False)
        self.optimizer = optim.SGD([{'params':model.bird_model.parameters()},
                            {'params':model.rn50_model.avgpool.parameters()},
                            {'params':model.rn50_model.layer4.parameters()}],
                                   lr=0.01, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma= 0.9)
        self.scheduler = None
        self.trainer = Trainer(model, self.device, self.traindl, valdl, self.criterion, self.optimizer, self.scheduler, writer=None)
        return self.trainer


# Uses Iterative data set, no transforms. mostly for speed tests
class Expt_T1i(Expt):
    def __init__(self,db,args):
        trainTrans = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.AutoAugment(),
            #transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=.3),
            # transforms.RandomRotation(20),
            #    transforms.CenterCrop(224),
        ])
        super().__init__(db,args,trainTrans)
        self.criterion = nn.CrossEntropyLoss()



    def buildTrainer(self,model):
        traindb = self.db.getTrainDB()
        useval = self.args[3]
        valds = None
        if useval:
            valds = self.db.getValDB()
        self.trainds = IterImageset(traindb,batchsize=self.batch_size,todev=self.todev)
        self.traindl = self.trainds  #.makeDataloader()
        self.optimizer = optim.SGD([{'params':model.bird_model.parameters()},
                            {'params':model.rn50_model.avgpool.parameters()},
                            {'params':model.rn50_model.layer4.parameters()}],
                                   lr=0.01, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma= 0.9)
        self.scheduler = None
        self.trainer = Trainer(model, self.device, self.traindl, valds, self.criterion, self.optimizer, self.scheduler, writer=None)
        return self.trainer



