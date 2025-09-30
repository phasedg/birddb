
###
### This is a  holder for directories and device etc
###  Usually a singleton
###
class Env:

  TheEnv = None
  @staticmethod
  def initEnv(basedir,datadir,device):
    Env.TheEnv = Env(basedir,datadir,device)


  def __init__(self,basedir,datadir,device):
    self.device = device
    self.basedir = basedir
    self.modeldir = f"{basedir}/models"  # model files
    self.expdir = f"{basedir}/trainrun"  # training data
    self.datadir = datadir  # databases
    self.statdir = f"{basedir}/stats"  # test results
