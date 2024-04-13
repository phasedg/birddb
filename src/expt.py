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

'''
  Expt subclasses encapsule a model structure, training regime
  Instances add training dataset, and paths
'''
class Expt:
    def __init__(self,expdir,datadir,dbname,device,trainTrans):
        self.datadir = datadir
        self.dbname = dbname
        self.dbdir = Path(datadir / dbname)
        self.expdir = expdir
        if not os.path.exists(self.expdir):
            os.makedirs(self.expdir)

        self.device = device
        self.chkpath = self.expdir + '/check.pt'
        self.model = None
        self.trainTrans = trainTrans
        self.trainds = birddb.TrainImageDataset(self.dbdir,transform=self.trainTrans)
        self.trainds.name = f'Train_{dbname}'
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
    
    def trainModel(self,epochs,write=True,dotfreq=10):
        self.trainer = self.buildTrainer()
        self.trainer.dotfreq=dotfreq
        if self.hasCheckpoint():
            self.trainer.restore(self.chkpath)
        losses, accs = self.trainer.train(epochs,self.callback)
        self.trainer.model.writeModel(self.expdir,'model.pt')
        df = pd.DataFrame({'loss': losses, 'acc': accs})
        df.to_csv(str(self.expdir) + '/train_acc.csv',index=False)



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
        df.to_csv(expbase + '/stats.csv')
        return df


class Expt_RN50_V1_T2(Expt):
    def __init__(self,expbase,datadir,dbname,device):
        trainTrans = transforms.Compose([
            transforms.AutoAugment(),
            #transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=.3),
            # transforms.RandomRotation(20),
            #    transforms.CenterCrop(224),
        ])
        self.expbase = expbase
        super().__init__(str(expbase) + '/RN50_T1/T2/'+ dbname,datadir,dbname,device,trainTrans)
        self.criterion = nn.CrossEntropyLoss()



    def buildTrainer(self):
        self.traindl = self.trainds.makeDataloader(batch_size=32,shuffle=True)
        self.model = models.RN50_Bird_V1('RN50_V1_' + self.trainds.name,self.trainds.numClasses())
        self.optimizer = optim.SGD([{'params':self.model.bird_model.parameters()},
                            {'params':self.model.rn50_model.avgpool.parameters()},
                            {'params':self.model.rn50_model.layer4.parameters()}],

                                   lr=0.01, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma= 0.9)
        self.trainer = Trainer(self.model, self.device, self.traindl, None, self.criterion, self.optimizer, self.scheduler, writer=None)
        return self.trainer



if __name__ == "__main__":
    datadir = Path('/data1/datasets/birds')
    dbname = 'CUB_wood_sm'
    expbase = '/home/dg/proj/birddb/expts'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    expt = Expt_RN50_V1_T2(expbase,datadir,dbname,device)
    print(expt.dirDict())
    expt.trainModel(5)

    df = pd.read_csv(expt.expdir + '/train_acc.csv')
    print(df)
    df.plot()
    exit()
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
