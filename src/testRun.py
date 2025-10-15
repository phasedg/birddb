import os
from collections import namedtuple

ImScore = namedtuple("ImScore",["ClassId", "Prob"])
class TestRun:

  def __init__(self,rundir,modname,moddb,tstname):
    self.modname = modname
    self.tstname = tstname
    self.moddb = moddb
    self.rundir = rundir
    self.fname = f"{rundir}/{tstname}-imscores.csv"
    self.runScores = {}
    print(self.fname)
    if os.path.exists(self.fname):
      self.load()

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
    print(f"{len(self.runScores)} Loaded")

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
      for v in self.runScores:
        f.write(v[0])
        for i in range(0,5):
          f.write(f",{v[1][i].ClassId},{v[1][i].Prob:.3f}")
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










