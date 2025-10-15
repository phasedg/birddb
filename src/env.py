import os
import torch
###
### This is a  holder for directories and device etc
###  Usually a singleton
###
class Env:

  TheEnv = None
  @staticmethod
  def initEnv(basedir,datadir,device):
    Env.TheEnv = Env(basedir,datadir,device)
  
  @staticmethod
  def setupEnv():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wd = os.getcwd()
    if "proj/birddb" in wd:
      datadir = '/data1/datasets/birds'
      basedir = "/home/dg/proj/birddb"
      Env.initEnv(basedir,datadir,device)
    if "studio" in wd: # lightning.ai
      basedir = f"{wd}/birddb"
      datadir = f"{basedir}/data"
      Env.initEnv(basedir,datadir,device)



  def __init__(self,basedir,datadir,device):
    self.device = device
    self.basedir = basedir
    self.modeldir = f"{basedir}/models"  # model files
    self.expdir = f"{basedir}/trainrun"  # training data
    self.datadir = datadir  # databases
    self.statdir = f"{basedir}/stats"  # test results
    self.srcdir = f"{basedir}/src"
