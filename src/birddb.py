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
from PIL import Image

RESNET_IMSIZE = 224

'''
Base class for image data set 
'''
class ImageDataset(Dataset):
    RESIZE_TRANS = transforms.Compose([
        transforms.Resize([RESNET_IMSIZE,RESNET_IMSIZE],antialias=True),
        transforms.ToDtype(torch.float32,scale=True),
    ])

    def __init__(self, cubdir, pred=None, transform=None, traincode = '1'):
        self.dir = cubdir
        self.img_dir = cubdir / 'images'
        self.transform = transform
        self.traincode = traincode
        
        self.myClassLabs = self.readClassLabs(cubdir) # [[lab, name]]
        self.classLabDict = { x[0]:x[1] for x in self.myClassLabs} # text label to name dict
        self.myClassLabs = [x[0] for x in self.myClassLabs if (pred is None) or pred(x[1])] # filtered list maintaining order

        # now we need to filter image list based on classes
        self.imageLabDict = {x[0]:x[1] for x in self.readImageLabs(cubdir)} # [[id lab]] convert to dict
        self.myIDs = set([x[0] for x in self.readTrainTest(cubdir) if (traincode == 'all' or x[1] == traincode) and (self.imageLabDict[x[0]] in self.myClassLabs)]) # filter ids on train/test and classlist; set for quick access 
        self.usedLabs = set(self.imageLabDict[x] for x in self.myIDs) # not all classes used -- ie NAB superclasses
        self.myClassLabs = [x for x in self.myClassLabs if x in self.usedLabs] # filtered by those in use -- list maintaining original order        
        self.myClassNames = [self.classLabDict[x] for x in self.myClassLabs] # corresponding class names
        self.imageDict = {}
        self.imageLabs = []
        self.imageFiles = []
        self.name = None

    def numClasses(self):
        return len(self.myClassLabs)

    def __len__(self):
        return len(self.imageFiles)

    def __getitem__(self, idx):
        if idx in self.imageDict:
          image = self.imageDict[idx]
        else:
          img_path = os.path.join(self.img_dir, self.imageFiles[idx])
          image = read_image(img_path,ImageReadMode.RGB) #some CBU in BW
          self.imageDict[idx] = image
         # print(f'ImageDict = {len(self.imageDict)}')
        label = self.imageLabs[idx]
        if self.transform:
            image = self.transform(image)
        if image.shape[0] < 3:
            print(image.shape,self.myFiles[idx])
        return image, label

    def getClassName(self,lab):
        return self.myClassNames[lab]

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

        
    def makeDataloader(self,batch_size,shuffle=False):
        if self.transform is None:
            self.transform = ImageDataset.RESIZE_TRANS
        else:
            self.transform = transforms.Compose([self.transform,ImageDataset.RESIZE_TRANS ])           
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
 Train Dataset potentially filtered
'''
class TrainImageDataset(ImageDataset):
    def __init__(self,cubdir, pred=None, transform=None,trainset='train'):
        super().__init__(cubdir,pred,transform,'1' if trainset == 'train' else ('0' if trainset == 'test' else trainset))
        # dont assume same order -- preserve file list order
        for x in self.readImageFiles(cubdir): #[[id, file]]):
            myid = x[0]
            mylab = self.imageLabDict[myid]
            if myid in self.myIDs:
                self.imageFiles.append(x[1]) # imagefile
                self.imageLabs.append(self.myClassLabs.index(mylab)) # convert to index 
#        print(f'{len(self.imageFiles)} Images, {len(self.imageLabs)} Labels, {self.imageFiles[0]}')


'''
Test data set. Filtered by pred on className, classLabs and className arrays are from Training Dataset (tds) for consistancy with model
'''
class TestImageDataset(ImageDataset):
    def __init__(self, cubdir, modelds, trainset='test', pred=None, transform=None):
        super().__init__(cubdir,pred,transform,'1' if trainset == 'train' else '0')
        # now we map to labels/indexes in modelds, filtering images if needed
        # dont assume same order -- preserve file list order
        self.usedLabs = self.usedLabs.intersection(modelds.usedLabs)  # labels must be in model classLabs
        for x in self.readImageFiles(cubdir): #[[id, file]]):
            myid = x[0]
            mylab = self.imageLabDict[myid]
            if myid in self.myIDs and mylab in self.usedLabs:
                self.imageFiles.append(x[1]) # imagefile
                self.imageLabs.append(modelds.myClassLabs.index(mylab)) # convert to index 
        self.myClassLabs = [x for x in modelds.myClassLabs] # copy
        self.myClassNames = [x for x in modelds.myClassNames] # copy
#        print(f'{len(self.imageFiles)} Images, {len(self.imageLabs)} Labels, {self.imageFiles[0]}')



'''
 Converts a bird dataset for testing on a model trained on <modelds>

  Start with normal filtered dataset, Then convert classids and names to modelds (filtering where necessary)
  match on cleaned names

'''
class CrossImageDataset(ImageDataset):
    def __init__(self, cubdir, modelds, trainset = 'train',  pred=None, transform=None, target_transform=None):
        super().__init__(cubdir,pred,transform,'1' if trainset == 'train' else '0')  
        # now match cleaned names
        self.clNameDict= {}
        # clean model names
        for i,x in enumerate(modelds.myClassLabs):
            self.clNameDict[self.cleanModelName(modelds.myClassNames[i])] = x  # map cleaned names to model class ids
        self.clLabMap = {}  # NAB class lab to CBU class lab where map allowed
        for i,x in enumerate(self.myClassNames):
            cx = self.cleanTestName(x)
            if cx in self.clNameDict:
                self.clLabMap[self.myClassLabs[i]]  = self.clNameDict[cx]

        # dont assume same order -- preserve file list order
        self.imageLabs = []
        self.imageFiles = []
        for x in self.readImageFiles(cubdir): #[[id, file]]):
            myid = x[0]
            mylab = self.imageLabDict[myid]
            if myid in self.myIDs and mylab in self.clLabMap:
                self.imageFiles.append(x[1])
                mymodlab = self.clLabMap[mylab]
                self.imageLabs.append(modelds.myClassLabs.index(mymodlab)) # convert to index
        self.myClassLabs = [x for x in modelds.myClassLabs] # copy
        self.myClassNames = [x for x in modelds.myClassNames] # copy
    #    print(f'{len(self.imageFiles)} Images, {len(self.imageLabs)} Labels, {self.imageFiles[0]}')

        # note- strips () subcats in test to map to base in model
    def cleanTestName(self,c):
        if '(' in c:
            c = c[0:c.index('(')]
            c = c.strip()
        c = self.cleanModelName(c)
        return c

        # note- keeps () subcats in NAB
    def cleanModelName(self,c):
        if '.' in c:
            c = c.split('.')[1]
        c = c.replace('-','_').replace(' ','_').replace(',','').lower()
        return c



__trainTrans__ = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=.3),
    transforms.RandomRotation(20),
    #    transforms.CenterCrop(RESNET_IMSIZE),
])

def __testDataset__(datadir, dbname,test=None):
    cubdir = Path(datadir / dbname)
    CTrds = TrainImageDataset(cubdir)
    CTrds.name = f'Train Dataset from {dbname} - no pred'
    if test is None or test == 0:
        CTrds.printStats()
        CTrds.showImages()
    CTeds = TestImageDataset(cubdir,CTrds,'test')
    CTeds.name = f'Test Dataset from {dbname} - no pred'
    if test is None or test == 1:
        CTeds.printStats()
        CTeds.showImages()
    pred = lambda x: ('Woodpecker' in x) or ('Sapsucker' in x)
    CTeds = TestImageDataset(cubdir,CTrds,pred = pred)
    CTeds.name = f'Test Dataset from {dbname} - woodpeckers'
    if test is None or test == 2:
        CTeds.printStats()
        CTeds.showImages()
    WCTrds = TrainImageDataset(cubdir,pred = pred)
    WCTrds.name = f'Train Dataset from {dbname} - woodpeckers'
    if test is None or test == 3:
        WCTrds.printStats()
        WCTrds.showImages()
    CTeds = TestImageDataset(cubdir,WCTrds)
    CTeds.name = f'Test Dataset from {dbname} - woodpeckers from Train'
    if test is None or test == 4:
        CTeds.printStats()
        CTeds.showImages()

    if test is None or test == 5:
        CTrds.showLoaderImages()
    if test is None or test == 6:
        CTrds.showLoaderImages(shuffle=True)
    if test is None or test == 7:
        CTeds.showLoaderImages(shuffle=True)

    WCTrds.transform = __trainTrans__
    WCTrds.name = f'Train Dataset from {dbname} - woodpeckers - trainTrans'
    if test is None or test == 8:
        WCTrds.printStats()
        WCTrds.showImages()
    if test is None or test == 9:
        WCTrds.showLoaderImages(shuffle=True)


if __name__ == "__main__":
    datadir = Path('/data1/datasets/birds')
    
    nabdir = Path(datadir / 'nabirds')
    wdir = Path(datadir / 'nabirds_bbs')
    cubdir = Path(datadir / 'CUB_200_2011')
    pred = lambda x: ('Woodpecker' in x) or ('Sapsucker' in x)
    pred = None
    woodds = TrainImageDataset(nabdir,pred,None,'all')
    woodds.writeDBfiles(str(wdir))
    exit()
    __testDataset__(datadir, 'CUB_200_2011_bbs')
    exit()

    __testDataset__(datadir, 'CUB_200_2011')
    __testDataset__(datadir, 'nabirds')
    __testDataset__(datadir, 'nabirds_sm')

    exit()
   

#    cubds.showStats()
#    testds(cubds)
    copySmall(cubdir)
    exit()
#    testds(nabds)

    wcubds = TestImageDataset(cubdir,cubds,'test',pred)
    testds(wcubds)
    cds = CrossImageDataset(nabdir,cubds,'test',pred)
    testds(cds)

 #   testds(wcubds)
#    wnabds = TrainImageDataset(nabdir,pred,None,'all')
 #   testds(wnabds)
#    woodds = CrossImageDataset(Path('./wood'),cubds)
#    testds(woodds)    
