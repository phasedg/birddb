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
from PIL import Image, ImageDraw
from collections import namedtuple
from env import Env
import random


BdImage = namedtuple("BdImage",["Id","TrainTest","ClassId","ImFile","ClassIdx"])
BdClass = namedtuple("BdClass",["ClassId","ClassName","Index","Parent","CleanName","Tags"])

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
        if sname == "cub_bs":
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
            cname = x[1].strip()
            clean , tags = self.cleanName(cname)
            self.classes.append(BdClass(x[0],x[1].strip(),i,None,clean,tags))

    def loadABAData(self,aba):
        fname = f"{self.dbdir}/ABA_syn.txt"
        syns = {}
        if os.path.exists(fname):
          with open(fname,'r') as f:
            for line in f.readlines():
              if line[0] == '#':
                continue
              fields = line.split(',')
              cn = fields[0].strip()
              syn = [s.strip() for s in fields[1:]]
              syns[cn] = syn
        print(f'{len(syns)} Syns loaded')
        print(syns)
       
        abadict = {}
        for cl in self.classes:
          match = aba.match(cl.CleanName)
          if match is None and cl.CleanName in syns:
              match = aba.match(syns[cl.CleanName][0]) # for now pick first -- ok for family mapping
          if match is not None:
              abadict[cl.ClassId] = match
          else:
              print(f'no match for {cl.CleanName}')
        return abadict
    
    def famClassDict(self,dict):  # maps famname to index, also classid to fam index/id
        fs = set()
        for cl in self.classes:
          if cl.ClassId in dict:
            fs.add(dict[cl.ClassId].TaxonFamily)
        fl = list(fs)
        fl = sorted(fl)
        fd = {n:i for i,n in enumerate(fl)}
        cd = {}
        for cl in self.classes:
          if cl.ClassId in dict:
            tf = dict[cl.ClassId].TaxonFamily
            cd[cl.ClassId] = fd[tf]

        return fd,cd

    def loadBBData(self):
        bbdict = {}
        fname = f"{self.dbdir}/bounding_boxes.txt"
        with open(fname,'r') as f:
          for i,l in enumerate(f.readlines()):
            fields = l.split(' ')
            bbdict[fields[0]] = (float(fields[1]),float(fields[2]),float(fields[3]),float(fields[4]))
        return bbdict
    
    def loadBBDataFromFile(self,bbf):
        bbdict = {}
        fname = f"{self.dbdir}/{bbf}.txt"
        with open(fname,'r') as f:
          for i,l in enumerate(f.readlines()):
            fields = l.split(' ')
            bbdict[fields[0]] = (float(fields[1]),float(fields[2]),float(fields[3]),float(fields[4]))
        return bbdict
         
  
    # we write all yolo label files and use .txt file for train/val/test
    def writeYoloLabelFiles(self,cdict):
        bbdict = self.loadBBData()
        labdir = self.imdir.replace('images','labels') # must be only one instance
        if not os.path.isdir(labdir):
            os.mkdir(labdir)
        # write image list
        for id in self.imdata:
          if id.ClassId not in cdict:
              continue
          cidx = cdict[id.ClassId]
          imfile = f'{self.imdir}/{id.ImFile}'
          im = Image.open(imfile)
          iw = im.width
          ih = im.height
          fname = f'{labdir}/{id.ImFile.replace('jpg','txt')}'
          if not os.path.isdir(os.path.dirname(fname)):
            os.mkdir(os.path.dirname(fname))
          
          bb = bbdict[id.Id]
          with open(fname,'w') as f:
              f.write(f'{cidx} {(bb[0]+(bb[2]/2))/iw} {(bb[1] + (bb[3]/2))/ih} {bb[2]/iw} {bb[3]/ih}\n')  ## yolo wants cx,cy,w,h, bb is cornet w,h -- normalized retarded again

    def writeYoloImageFiles(self,cdict,fdir):  # yolo filesin txt must be absolute paths????? retarded
        train = []
        val = []
        test = []
        for id in self.imdata:
          if id.ClassId not in cdict:
              continue
          fname = f'{fdir}/images/{id.ImFile}'
          if id.TrainTest == '1':
              train.append(fname)
          if id.TrainTest == '0':
            test.append(fname)
            if random.random() < 0.1:
                val.append(fname)
        fname = f'{self.dbdir}/yoloTrain.txt'
        with open(fname,'w') as f:
          for l in train:
            f.write(l)
            f.write('\n')
        fname = f'{self.dbdir}/yoloTest.txt'
        with open(fname,'w') as f:
          for l in test:
            f.write(l)
            f.write('\n')
        fname = f'{self.dbdir}/yoloVal.txt'
        with open(fname,'w') as f:
          for l in val:
            f.write(l)
            f.write('\n')

    def writeYoloYamlFile(self,ndict):
        fname = f'{self.dbdir}/famYolo.yaml'
        with open(fname,'w') as f:
           f.write(f'path: {self.dbdir}\n')
           f.write(f'train: yoloTrain.txt\n')
           f.write(f'val: yoloVal.txt\n')
           f.write(f'test: yoloTest.txt\n')
           f.write(f'names:\n')
           for c in ndict:
              f.write(f'  {ndict[c]}: {c}\n')
        

        
    
            
            
        
        

    

        
           

    def numImages(self):
        return len(self.imdata)
    
    def __len__(self):
        return len(self.imdata)
    
    def numClasses(self):
        return len(self.classes)
    
    def className(self,idx):
        return self.classes[idx].ClassName
    
    def classCounts(self):
        counts = {}
        for im in self.imdata:
            if im.ClassId not in counts:
                counts[im.ClassId] = 0
            counts[im.ClassId] += 1
        return counts
    
    def classHist(self):
        hist = {}
        for id,count in self.classCounts().items():
            if count not in hist:
                hist[count] = 0
            hist[count] += 1
        return hist
    
    def cleanName(self,n):
        if '.' in n:
            n = n[n.index('.')+1:]
        
        tags = None
        if '(' in n:
            tags = n[n.index('(')+1:n.index(')')]
            n = n[0:n.index('(')].strip()
        n = n.replace(' ','_').replace('-','_').lower()     
        return n, tags
    
    def classId(self,idx):
        return self.classes[idx].ClassId
    
    def imageId(self,idx):
        return self.imdata[idx].Id
    
    def getClassIdForImId(self,imid):
        if len(self.imdict) == 0:
            for im in self.imdata:
                self.imdict[im.Id] = im
        return self.imdict[imid].ClassId
    
    def classNameForClassId(self,cid):
        for cl in self.classes:
            if cl.ClassId == cid:
                return cl.ClassName
        return None
        

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
    
    def imageFileFromImid(self,imid):
        if len(self.imdict) == 0:
            for im in self.imdata:
                self.imdict[im.Id] = im
        return f"{self.imdir}/{self.imdict[imid].ImFile}"
        
        
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
    
    def sizeStats(self):
        bms = []
        for idata in self.imdata:
            pcount = 0
            rsum = 0
            gsum = 0
            bsum = 0
            fname = f"{self.imdir}/{idata.ImFile}"
            img = Image.open(fname)
            bms.append((img.width,img.height))
        avew = sum([b[0] for b in bms])/len(bms)
        aveh = sum([b[1] for b in bms])/len(bms)
        return avew,aveh,bms
    
    def showImages(self,cid):
        imfiles = [self.imageFileFromImid(im.Id)  for im in self.imdata if im.ClassId == cid]
        fig, axes = plt.subplots(nrows=6, ncols=5,figsize=(15,18))
        imfiles = imfiles[0:30]
        for i, imfile in enumerate(imfiles):
            img = Image.open(imfile)
            axes[i//5,i%5].imshow(img)
        fig.suptitle(f'{self.classNameForClassId(cid)}')
        plt.show()

    def showImageWithBox(self,imid,bbfile):
        imfile = self.imageFileFromImid(imid)
        fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(6,6))
        img = Image.open(imfile)
        imdraw = ImageDraw.Draw(img)
        
        bbdict = self.loadBBDataFromFile(bbfile)
        bbdata = bbdict[imid]
        imdraw.rectangle([bbdata[0],bbdata[1],bbdata[0]+bbdata[2], bbdata[1]+bbdata[3]],outline=(200,0,0),width=2)

        axes.imshow(img)
        fig.suptitle(f'{imid}')
        plt.show()

RESNET_IMSIZE = 224

'''
PLaceholder for older stuff
'''
class ModifiedBirdDB(BirdDB):
    

    def __init__(self, basedb,dbname,bbfile=None):
    
      self.basedb = basedb
      self.dbname = dbname
      # shallow copy to start
      self.classes = basedb.classes
      self.imdata = basedb.imdata
      self.imdict = {}
      for im in self.imdata:
                self.imdict[im.Id] = im
      self.dir = db.dir
      self.dbdir = f"{self.dir}/{dbname}"
      self.imdir = self.dbdir + '/images'
      if self.dbname.endswith('_bby'):
         bbfile = 'bounding_boxes_yolo11'
      if self.dbname.endswith('_bb4'):
         bbfile = 'bounding_boxes_yolo_e40'
      if bbfile is not None:
        self.bbdict = db.loadBBDataFromFile(bbfile)
  
    def writeDBfiles(self):
      if not os.path.exists(self.dbdir):  # make possible subdirs
            os.makedirs(self.dbdir)
      self.writeClassLabs()
      self.writeImageLabs()
      self.writeTrainTest()
      self.writeImageList()
      if '_bb' in self.dbname:
        self.writeBBImageFiles()
       

    
    def writeClassLabs(self):
        with open(self.dbdir + '/classes.txt','w') as f:
          for x in self.classes:
              f.write(f'{x.ClassId} {x.ClassName}\n')


    def writeImageLabs(self):
        with open(self.dbdir + '/image_class_labels.txt','w') as f:
          for x in self.imdata:
            f.write(f'{x.Id} {x.ClassId}\n')

    def writeTrainTest(self):
        with open(self.dbdir + '/train_test_split.txt','w') as f:
          for x in self.imdata:
            f.write(f'{x.Id} {x.TrainTest}\n')

    def writeImageList(self):
        with open(self.dbdir + '/images.txt','w') as f:
          for x in self.imdata:
            f.write(f'{x.Id} {x.ImFile}\n')
     
    def copyImageFile(self,imfile):
        
        fname = f'{self.imdir}/{imfile}'
        sname = f'{self.basedb.imdir}/{imfile}'
        fdir = os.path.dirname(fname)
        if not os.path.exists(fdir):  # make possible subdirs
            os.makedirs(fdir)
        shutil.copy(sname, fname)


    def copyImageFiles(self):
        if not os.path.exists(self.imdir):
            os.makedirs(self.imdir)
        for x in self.imdata:
           self.copyImageFile(x.ImFile)

    def writeBBImageFiles(self):
        if not os.path.exists(self.imdir):
            os.makedirs(self.imdir)
        for x in self.imdata:
           self.writeBBFile(x.Id)

    def writeBBFile(self,imid):
      idata = self.imdict[imid]
      imfile = idata.ImFile
      fname = f'{self.imdir}/{imfile}'
      sname = f'{self.basedb.imdir}/{imfile}'
      im = Image.open(sname)
      bb = self.bbdict[imid]
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
      if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
      print(f"writing {fname}")
      im.save(fname)     

        

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
    #db = BirdDB.DBFromName('cub_os')
    #ndb = ModifiedBirdDB(db,'cub_bb4')
    #ndb.writeDBfiles()
    #exit()

    db = BirdDB.DBFromName('cub_bb4')
    #for l in db.imdata[0:10]:
    db.showImages('166')
    exit()
    
    from abaList import ABAList
    aba = ABAList()
   
    db = BirdDB.DBFromName("cub_os")
    adict = db.loadABAData(aba) 
    fdict, cdict = db.famClassDict(adict)
   # db = BirdDB.DBFromName("cub_os")
    print(cdict)
    db.writeYoloLabelFiles(cdict)
    #db.writeYoloImageFiles(cdict,'/teamspace/studios/this_studio/birddb/data/CUB_200_2011')
   
    #db.writeYoloYamlFile(fdict)
    exit()

    aba = db.loadABAData(aba)
    print(aba[db.classes[0].ClassId])
    print(db.famClassDict()[0])
    print(db.famClassDict()[1])
    print(db.imdata[0])
    exit()
    exit()

    tag = '1'
    db = BirdDB.DBFromName("cub_sm")
    db.getTrainDB().showImages(tag)
    db.getTestDB().showImages(tag)
    exit()
   # db = db.getTrainDB()
    print(db.classes[0])
    print(db.imdata[0])
    print(db.imdata[1000])
   # print(db.pixelStats())
    print(db.classHist())
    count = 0
    for k,c in db.classHist().items():
        count += k*c
    print(count,len(db.classCounts()))
    cnts = db.classHist()
    x = [i for i in cnts]
    y = [cnts[i] for i in cnts]
    from matplotlib import pyplot as plt
    plt.bar(x,y)
    plt.show()
        
    exit()
   
   