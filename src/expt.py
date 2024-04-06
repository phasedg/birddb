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

'''
  Expt subclasses encapsule a model structure, training regime
  Instances add training dataset, and paths
'''
class Expt:
    def __init__(self):
        pass

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
        with torch.no_grad():
            # Iterate over data.
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

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



class Expt_RN50_V1_T1:
    def __init__(self,expdir,datadir,dbname,device):
        self.datadir = datadir
        self.dbname = dbname
        self.dbdir = Path(datadir / dbname)
        self.trainTrans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=.3),
            # transforms.RandomRotation(20),
            #    transforms.CenterCrop(224),
        ])

        self.trainds = birddb.TrainImageDataset(self.dbdir,transform=self.trainTrans)
        self.trainds.name = f'Train_{dbname}'
        self.traindl = self.trainds.makeDataloader(batch_size=32,shuffle=True)
        self.expdir = expdir
        self.device = device


        self.model = models.RN50_Bird_V1('RN50_V1' + '-' + self.trainds.name,self.trainds.numClasses())
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD([{'params':self.model.bird_model.parameters()},
                            {'params':self.model.rn50_model.avgpool.parameters()},
                            {'params':self.model.rn50_model.layer4.parameters()}],

                                   lr=0.01, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma= 0.5)
        self.trainer = Trainer(self.model, self.device, self.traindl, None, self.criterion, self.optimizer, self.scheduler, writer=None,dotfreq=0, chkpath=self.expdir + '/check.pt')

    def trainModel(self,epochs,write=True):
        losses, accs = self.trainer.train(epochs)
        self.model.writeModel(self.expdir,'model.pt')

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
        with torch.no_grad():
            # Iterate over data.
            for inputs, labels in dl:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

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

    
class Expt_RN50_V1_T2(Expt):
    def __init__(self,expdir,datadir,dbname,device):
        super().__init__()
        self.datadir = datadir
        self.dbname = dbname
        self.dbdir = Path(datadir / dbname)
        self.trainTrans = transforms.Compose([
            transforms.AutoAugment(),
            #transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=.3),
            # transforms.RandomRotation(20),
            #    transforms.CenterCrop(224),
        ])

        self.trainds = birddb.TrainImageDataset(self.dbdir,transform=self.trainTrans)
        self.trainds.name = f'Train_{dbname}'
        self.traindl = self.trainds.makeDataloader(batch_size=32,shuffle=True)
        self.expdir = expdir
        self.device = device


        self.model = models.RN50_Bird_V1('RN50_V1_' + '-' + self.trainds.name,self.trainds.numClasses())
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD([{'params':self.model.bird_model.parameters()},
                            {'params':self.model.rn50_model.avgpool.parameters()},
                            {'params':self.model.rn50_model.layer4.parameters()}],

                                   lr=0.01, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma= 0.9)
        self.trainer = Trainer(self.model, self.device, self.traindl, None, self.criterion, self.optimizer, self.scheduler, writer=None,dotfreq=0, chkpath=self.expdir + '/check.pt')

    def trainModel(self,epochs,write=True):
        losses, accs = self.trainer.train(epochs)
        self.model.writeModel(self.expdir,'model.pt')
        df = pd.DataFrame({'loss': losses, 'acc': acc})
        df.to_csv(str(self.expdir) + '/train_acc.csv')



if __name__ == "__main__":
    datadir = Path('/data1/datasets/birds')
    dbname = 'CUB_wood_sm'
    expdir = str(Path('/home/dg/proj/birddb/expts/RN50_V1_' + dbname))
    if not os.path.exists(expdir):
        os.makedirs(expdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    expt = Expt_RN50_V1_T1(expdir,datadir,dbname,device)

#    expt.trainModel(10)
#    exit()
    dbdir = Path(datadir / dbname)

    model = expt.loadModel()
    # note - makeDataloader always adds resize transform
    trainds = birddb.TrainImageDataset(dbdir)
    testds = birddb.TestImageDataset(dbdir,trainds)
    

#    expt.visualizeOnDataset(trainds)
#    expt.visualizeOnDataset(testds)


    dbname = 'CUB_wood'
    dbdir = Path(datadir / dbname)    
    testds = birddb.TestImageDataset(dbdir,trainds)
    testds.printStats()
    expt.testOnDataset(testds)
