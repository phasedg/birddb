import os
from env import Env
from collections import namedtuple
import re

ABAData = namedtuple("ABAData",["Family","TaxonFamily","CommonName","CleanName","Binomial","Code"])

class ABAList:

  def __init__(self,abaFile='ABA_Checklist-8.13a.csv'):
    Env.setupEnv()
    self.abaFile = f"{Env.TheEnv.datadir}/{abaFile}"
    self.abalist = []
    self.dictByClean = {}
    self.load()

  def load(self):
    if len(self.abalist) > 0:
      return
    fam = None
    with open(self.abaFile,'r') as f:
      for i, line in enumerate(f.readlines()):
        
        if i <2:  # comment line
          continue
        if line.startswith(",,,"):
          continue
        if line[0] == '"':
          ix = line.index('"',1)
          fam = line[1:ix].replace('"','')
          continue
        fields = line.split(',')
        
      
        
       
        comname = fields[1]
        binomial = fields[3]
        code = fields[4]
        cleanname = comname
        if '(' in cleanname:
          cleanname = cleanname[0:cleanname.index('(')].strip()
        if '[' in cleanname:
          cleanname = cleanname[0:cleanname.index('[')].strip()
        cleanname = cleanname.replace('-','_').replace(' ','_').lower()
        ipar = fam.index('(')
        epar = fam.index(')')
        efam = fam[0:ipar].strip()
        lfam = fam[ipar+1:epar].strip()
        data = ABAData(efam,lfam,comname,cleanname,binomial,code)
        self.abalist.append(data)
        self.dictByClean[data.CleanName] = data
        if "'s" in data.CleanName: # inconsistant naming in DB
          self.dictByClean[data.CleanName.replace("'s",'')] = data
          self.dictByClean[data.CleanName.replace("'s",'s')] = data


  def match(self,cname):
    if cname in self.dictByClean:
      return self.dictByClean[cname]
    return None


if __name__ == "__main__":
  aba = ABAList()
  print(aba.abalist[0:4])


