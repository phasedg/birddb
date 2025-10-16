import torch
from torch.utils.data import Dataset
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
import random

BdImage = namedtuple("BdImage",["Id","TrainTest","ClassId","ImFile","ClassIdx"])
BdClass = namedtuple("BdClass",["ClassId","ClassName","Index","Parent"])

class BirdDB:
    
    @staticmethod
    def DBFromName(sname):
        
        dbname = sname
        if sname == "cub_os":
          dbname = "CUB_200_2011"
        if sname == "nab_os":
          dbname = "nabirds"
        if sname == "cub_sm":
          dbname = "CUB_200_2011_sm"
        if sname == "cub_bb":
          dbname = "CUB_200_2011_bbs"
        if sname == "cub_wood":
          dbname = "CUB_wood"
        if sname == "nab_sm":
          dbname = "nabirds_sm"
        if sname == "nab_wood_sm":
          dbname = "nab_wood_sm"
        if sname == "nab_bb":
          dbname = "nabirds_bbs"
        return BirdDB(sname,dbname)

    def __init__(self, sname, dbname):
        self.dir = Env.TheEnv.datadir
        self.dbname = dbname
        self.sname = sname
        self.dbdir = f"{self.dir}/{self.dbname}"
        self.imdir = f"{self.dbdir}/images"
        self.classes = []
        self.imdata = []
        self.imdict = {}
        self.loadFromFiles()
        self.means = None
        if "cub" in self.sname:
            self.means = [0.485,0.499,0.432]
        if "nab" in self.sname:
            self.means = [0.491,0.508,0.464]

    def loadFromFiles(self):
        if os.path.isdir(self.dbdir): # derived dbs don't have dirs
            self.loadClasses()
            self.loadImageData()
            print(f"{len(self.classes)} Classes, {len(self.imdata)} Images")

    def loadClasses(self):
        fname = f"{self.dbdir}/classes.txt"
        with open(fname,'r') as f:
          for i,l in enumerate(f.readlines()):
            x = l.split(' ',1) 
            self.classes.append(BdClass(x[0],x[1].strip(),i,None))

    def numImages(self):
        return len(self.imdata)
    
    def __len__(self):
        return len(self.imdata)
    
    def numClasses(self):
        return len(self.classes)
    
    def className(self,idx):
        return self.classes[idx].ClassName
    
    def classId(self,idx):
        return self.classes[idx].ClassId
    
    def imageId(self,idx):
        return self.imdata[idx].Id
    
    def getClassIdForImId(self,imid):
        if len(self.imdict) == 0:
            for im in self.imdata:
                self.imdict[im.Id] = im
        return self.imdict[imid].ClassId
        

    def getTrainDB(self):
        db = BirdDB(self.sname+"_trn",self.dbname)
        db.classes = self.classes
        db.imdata = [x for x in self.imdata if x.TrainTest == "1" ]
        print(f"{len(db.classes)} Classes, {len(db.imdata)} Images")
        return db
    
    def getTestDB(self):
        db = BirdDB(self.sname+"_tst",self.dbname)
        db.classes = self.classes
        db.imdata = [x for x in self.imdata if x.TrainTest == "0" ]
        print(f"{len(db.classes)} Classes, {len(db.imdata)} Images")
        return db
    
    def getValDB(self,frac = 0.2):
        db = BirdDB(self.sname+":ts",self.dbname)
        db.classes = self.classes
        db.imdata = [x for x in self.imdata if x.TrainTest == "0" and random.random() < frac ]
        print(f"{len(db.classes)} Classes, {len(db.imdata)} Images")
        return db
    
        
    
    def imageFileAndLabel(self,idx):
        d = self.imdata[idx]
        return f"{self.imdir}/{d.ImFile}", d.ClassIdx
        
    def loadImageData(self):
        cdict = {c.ClassId:i for i,c in enumerate(self.classes)}
        fname = f"{self.dbdir}/image_class_labels.txt"

        with open(fname,'r') as f:
            imageLabels = [l.split(' ',1) for l in f.readlines()]
            imageLabels = {x[0]:x[1].strip() for x in imageLabels} # read image labels to dict
        fname = f"{self.dbdir}/train_test_split.txt"
        with open(fname,'r') as f:
            trainTest = [l.split(' ',1) for l in f.readlines()]
            trainTest = {x[0]:x[1].strip() for x in trainTest} # read train/test to dict

        fname = f"{self.dbdir}/images.txt"
        with open(fname,'r') as f:
          for l in f.readlines():
              d = l.split(' ',1)
              data = BdImage(d[0],trainTest[d[0]],imageLabels[d[0]],d[1].strip(),cdict[imageLabels[d[0]]])  # could compress image name
              self.imdata.append(data)

    # averages per image, then averages averages (if images are same size should be global average)
    def pixelStats(self):
        rms = []
        gms = []
        bms = []
        for idata in self.imdata:
            pcount = 0
            rsum = 0
            gsum = 0
            bsum = 0
            fname = f"{self.imdir}/{idata.ImFile}"
            img = Image.open(fname)
            for x in range(img.width):
              for y in range(img.height):
                pixel = img.getpixel((x, y))
                rsum +=pixel[0]
                gsum +=pixel[1]
                bsum +=pixel[2]
                pcount += 255  # hack scaler
            rms.append(rsum/pcount)
            gms.append(gsum/pcount)
            bms.append(bsum/pcount)
        return sum(rms)/len(self.imdata), sum(gms)/len(self.imdata), sum(bms)/len(self.imdata)


RESNET_IMSIZE = 224

'''
PLaceholder for older stuff
'''
class MiscDB(Dataset):
    
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

    def __init__(self, db,  transform=None):
        
        self.transform = transform
        self.db = db
       
        
        self.imageDict = {}
        self.labelDict = {}
        
        self.name = None

    def numClasses(self):
        return self.db.numClasses()

    def __len__(self):
        return self.db.numImages();

    def __getitem__(self, idx):
        if idx in self.imageDict:
          image = self.imageDict[idx]
          label = self.labelDict[idx]
        else:
          img_path, label = self.db.imageFileAndLabel(idx)
          ## this reads into tensor
          image = torchvision.io.decode_image(img_path,ImageReadMode.RGB) #some CBU in BW
          image = ImageDataset.RESIZE_TRANS(image)  # resize and scale
          image = image.to(Env.TheEnv.device)
          self.imageDict[idx] = image
          self.labelDict[idx] = label
        # perhaps move to GPU here?
        # print(f'ImageDict = {len(self.imageDict)}')
       # image = ImageDataset.RESIZE_TRANS(image)  # resize and scale
        
        if False and self.transform:
          image = self.transform(image)  # possible run transform pipeline here
        if image.shape[0] < 3:
            print(image.shape,img_path)
        return image, label  # returns tensor and int

    def getClassName(self,idx):
        return self.db.className(idx)

    def writeClassLabs(self,wdir):
        f = open(wdir + '/classes.txt','w')
        for i, x in enumerate(self.myClassLabs):
            f.write(f'{x} {self.myClassNames[i]}\n')
        f.close()

    def writeImageLabs(self,wdir):
        f = open(wdir + '/image_class_labels.txt','w')
        for x in self.readImageFiles(self.dir): #[[id, file]]):
            myid = x[0]
            mylab = self.imageLabDict[myid]
            if myid in self.myIDs and mylab in self.usedLabs:
                f.write(f'{myid} {mylab}\n')
        f.close()

    def writeImageFiles(self,wdir):
        idir = wdir + '/images'
        if not os.path.exists(idir):
            os.makedirs(idir)
        f = open(wdir + '/images.txt','w')
        for x in self.readImageFiles(self.dir): #[[id, file]]):
            myid = x[0]
            mylab = self.imageLabDict[myid]
            if myid in self.myIDs and mylab in self.usedLabs:
                f.write(f'{myid} {x[1]}\n')
                iidir = idir + '/' + x[1].split('/')[0]
                if not os.path.exists(iidir):
                    os.makedirs(iidir)
                shutil.copy(self.dir / 'images' / x[1], idir + '/' + x[1])
        f.close()                

    def writeSmallImageFiles(self,wdir):
        f = open(wdir + '/images.txt','w')
        for x in self.readImageFiles(self.dir): #[[id, file]]):
            myid = x[0]
            mylab = self.imageLabDict[myid]
            if myid in self.myIDs and mylab in self.usedLabs:
                f.write(f'{myid} {x[1]}\n')
                img_path = str(self.dir / 'images' / x[1])
                write_path = wdir + '/images/' + x[1]
                iidir = wdir + '/images/' + x[1].split('/')[0]
                if not os.path.exists(iidir):
                    os.makedirs(iidir)
                # read image shink and write
                image = read_image(img_path,ImageReadMode.RGB) #some CBU in BW
                w = image.shape[1]
                h = image.shape[2]
                scale = 1
                if w < h and w > RESNET_IMSIZE:
                    scale = RESNET_IMSIZE/w
                    w = RESNET_IMSIZE
                    h = int(h*scale)
                elif h > RESNET_IMSIZE:
                    scale = RESNET_IMSIZE/h
                    h = RESNET_IMSIZE
                    w = int(w * scale)
                tr = transforms.Resize([w,h],antialias=True)
                image = tr(image)
                tu.write_jpeg(image,write_path,100)
        f.close()                

    '''
    Write images files cropping to bounding box (squared up where possible) then resized to 224,224 for ResNet.
    Uses PIL methods rather than torchvision (as in above) for simplicity.
    '''
    def writeBBSImageFiles(self,wdir):
        f = open(wdir + '/images.txt','w')
        bbdict = self.readBBDict(self.dir)
        for x in self.readImageFiles(self.dir): #[[id, file]]):
            myid = x[0]
            mylab = self.imageLabDict[myid]
            if myid in self.myIDs and mylab in self.usedLabs:
                f.write(f'{myid} {x[1]}\n')
                img_path = str(self.dir / 'images' / x[1])
                write_path = wdir + '/images/' + x[1]
                iidir = wdir + '/images/' + x[1].split('/')[0]
                if not os.path.exists(iidir):
                    os.makedirs(iidir)
                # read image shink and write
                im = Image.open(img_path)
                bb = bbdict[myid]
                left = bb[0]
                top = bb[1]
                w = bb[2]
                h = bb[3]
                right = left + w
                bottom = top + h
                pad = int(max(w,h) * 1.05) # pad by 5%
                pad = 0  # too much distortion?
                if w > h: # try to adjust height to match width plut pad
                    diff = int((w-h)/2) # amount to pad each side
                    top = max(0,bb[1] - diff)
                    bottom = min(im.height,top + w)
                else:
                    diff = int((h-w)/2)
                    left = max(0,bb[0]-diff)
                    right = min(im.width,left + h)
                left = max(0,left-pad)
                top = max(0,top-pad)
                right = min(im.width,right+pad)
                bottom = min(im.height,bottom+pad)
                im = im.crop((left,top,right,bottom))
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                
                im = im.resize((RESNET_IMSIZE,RESNET_IMSIZE))
                im.save(write_path)                     
        f.close()                


    def writeTrainTest(self,wdir):
        f = open(wdir + '/train_test_split.txt','w')
        ttdict = {x[0]:x[1] for x in self.readTrainTest(self.dir)} #[[id, cat]]):
        for x in self.readImageFiles(self.dir): #[[id, file]]):
            myid = x[0]
            mylab = self.imageLabDict[myid]
            if myid in self.myIDs and mylab in self.usedLabs:
                f.write(f'{myid} {ttdict[myid]}\n')
        f.close()                

    def writeDBfiles(self,wdir):
        if not os.path.exists(wdir):
            os.makedirs(wdir)
        self.writeClassLabs(wdir)
        self.writeImageLabs(wdir)
        if wdir.endswith('_sm'):
            self.writeSmallImageFiles(wdir)
        elif wdir.endswith('_bbs'):
            self.writeBBSImageFiles(wdir)
        else:
            self.writeImageFiles(wdir)
        self.writeTrainTest(wdir)        


    def printStats(self):
        print(f'{len(self.myClassNames)} Classes')
        print(f'{len(self.imageFiles)} Images')
        print(f'{len(self.imageLabs)} Labels')
        ccounts = {}
        for x in self.imageLabDict:
            ilab = self.imageLabDict[x]
            if ilab not in ccounts:
                ccounts[ilab] = 1
            else:
                ccounts[ilab] += 1
        mean = 0
        for x in ccounts:
            mean += ccounts[x]
        mean = mean/len(ccounts);
        print(f'Ave images per class: {mean}')

    def readClassLabs(self,d):
        with open(d / 'classes.txt','r') as f:
            classLabs = [l.split(' ',1) for l in f.readlines()]
            classLabs = [[x[0],x[1].strip()] for x in classLabs]
            return classLabs

    def readImageLabs(self,d):
        with open(d / 'image_class_labels.txt','r') as f:
            imageLabels = [l.split(' ',1) for l in f.readlines()]
            imageLabels = [[x[0],x[1].strip()] for x in imageLabels]
            return imageLabels

    def readTrainTest(self,d):
        with open(d / 'train_test_split.txt','r') as f:
            trainTest = [l.split() for l in f.readlines()]
            return trainTest

    def readImageFiles(self,d):
        with open(d / 'images.txt','r') as f:
            trainTest = [l.split() for l in f.readlines()]
            return trainTest

    def readBBDict(self,d):
        bbs = {}
        with open(d / 'bounding_boxes.txt','r') as f:
            for x in f.readlines():
                x = x.strip().split(' ')
                l = []
                for i in range(1,5):
                    l.append(int(x[i].split('.')[0]))
                bbs[x[0]] = l
        return bbs

        
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
            



if __name__ == "__main__":
    #__common_classes__()
    Env.setupEnv()
    
    db = BirdDB.DBFromName("nab_sm")
    db = db.getTrainDB()
    print(db.classes[0:5])
    print(db.imdata[0:5])
    print(db.pixelStats())
    
    exit()
   