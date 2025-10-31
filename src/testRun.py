import os
from collections import namedtuple
from matplotlib import pyplot as plt
from PIL import Image

ImScore = namedtuple("ImScore",["ClassId", "Prob"])
class TestRun:

  def __init__(self,rundir,modname,moddb,tstname):
    self.modname = modname
    self.tstname = tstname
    self.moddb = moddb
    self.rundir = rundir
    self.fname = f"{rundir}/{modname}/{tstname}-imscores.csv"
    self.runScores = {}
   # print(self.fname)
    if os.path.exists(self.fname):
      self.load()
    else:
      print(f"No File {self.fname}")

  def __len__(self):
    return len(self.runScores)

  def add(self,imId,imscores):  # imscores is list of top 5 tuples
    nscores = [ImScore(*x) for x in imscores] # convert to named tuple
    self.runScores[imId] = nscores
  
  def load(self):
    with open(self.fname) as f:
      for line in f.readlines():
        if line[0] == '#':
          continue
        fields = line.split(',')
        if fields[0] == "ImageId": # header
          continue
        imid = fields[0]
        scores = []
        for i in range(1,len(fields),2):
          scores.append((fields[i],float(fields[i+1])))
        self.add(imid,scores)
   # print(f"{len(self.runScores)} Loaded")

  def write(self):
    if not os.path.exists(os.path.dirname(self.fname)):
      os.makedirs(os.path.dirname(self.fname))
    with open(self.fname,'w') as f:
      f.write(f"#mod db: {self.moddb.sname}\n")
      f.write(f"#run db: {self.tstname}\n")
      f.write("ImageId")
      for i in range(0,5):
          f.write(f",Class{i},Prob{i}")
      f.write("\n")
      for imid, v in self.runScores.items():
        f.write(imid)
        for i in range(0,5):
          f.write(f",{v[i].ClassId},{v[i].Prob:.3f}")
        f.write("\n")

  def top1Acc(self,conf=0):
    corr=0
    count = 0
    for imid, scores in self.runScores.items():
      c = self.moddb.getClassIdForImId(imid)  # not right, need to map mod classes to ground truth test classs
      if scores[0].Prob < conf:
        continue  
      top1 = scores[0].ClassId
      if c == top1:
        corr += 1
      count += 1
    return corr/count, count
  
  def topNAcc(self,n):
    corr=0
    count = 0
    for imid, scores in self.runScores.items():
      c = self.moddb.getClassIdForImId(imid)  # not right, need to map mod classes to ground truth test classs
      for i in range(0,n):
        if c == scores[i].ClassId:
          corr += 1
      count += 1
    return corr/count, count
  
  def classAcc(self):
    scoredict = {c.ClassId:[0,0] for c in self.moddb.classes}
    for imid, scores in self.runScores.items():
      c = self.moddb.getClassIdForImId(imid)  # not right, need to map mod classes to ground truth test classs
      if c == scores[0].ClassId:
        scoredict[c][0] += 1
      scoredict[c][1] += 1
    tot = sum([s[0]/s[1] for k,s in scoredict.items()])
    ave = tot/self.moddb.numClasses()
    return ave, scoredict
  
  def visualizeErrors(self,cid):
    errors = []
    for imid, scores in self.runScores.items():
      c = self.moddb.getClassIdForImId(imid)  # not right, need to map mod classes to ground truth test classs
      if c != cid:
        continue
      if c != scores[0].ClassId:
        errors.append((imid,scores[0]))
    nerr = len(errors)
    if nerr == 0:
      print("No Errors!")
      return
    fig, axes = plt.subplots(nrows=1, ncols=nerr,figsize=(12,3))

    for i,err in enumerate(errors):
        if nerr == 1:
          ax = axes
        else:
          ax = axes[i]
        ax.axis('off')
        ax.set_title(f'label: {self.moddb.classNameForClassId(cid)} \npredicted: {self.moddb.classNameForClassId(err[1][0])}',fontsize=8)
        imfile = self.moddb.imageFileFromImid(err[0])
        img = Image.open(imfile)
        ax.imshow(img)
                    
    plt.show()

  def compareModels(self,modname):
    comptr = TestRun(self.rundir,modname,self.moddb,self.tstname)
    errdict = {}
    for imid, scores in self.runScores.items():
  
      prd = scores[0].ClassId
      c = self.moddb.getClassIdForImId(imid)  # not right, need to map mod classes to ground truth test class
      if c == prd:
        errdict[imid] = [0]
      else:
        errdict[imid] = [1]
    for imid, scores in comptr.runScores.items():
      prd = scores[0].ClassId
      c = self.moddb.getClassIdForImId(imid)  # not right, need to map mod classes to ground truth test class
      if c == prd:
        errdict[imid].append(0)
      else:
        errdict[imid].append(1)
    both = 0 ## wrong
    mod1 = 0
    mod2 = 0
    neith = 0
    for k,sc in errdict.items():
      if sc[0] == 1 and sc[1] == 1:
        both+=1
      if sc[0] == 1 and sc[1] == 0:
        mod1 += 1
      if sc[0] == 0 and sc[1] == 1:
        mod2 += 1
      if sc[0] == 0 and sc[1] == 0:
        neith += 1
    print(both,mod1,mod2,neith)

if __name__ == "__main__":
  from env import Env
  Env.setupEnv()
  from birddb import BirdDB
  mod1 = 'RN50v2_u1_e20_b32_l05_L4'
  mod2 = 'RX101v2_u1_e20_b32_l05_L4'
  rundb = BirdDB.DBFromName('cub_sm').getTestDB()
  tr = TestRun(f'./trainrun/cub_sm/',mod1,rundb,'cub_sm_tst')
  tr.compareModels(mod2)










