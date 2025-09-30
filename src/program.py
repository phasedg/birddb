import time
import os
import tempfile
import torch
import models
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import glob
from env import Env
from birddb import BirdDB
from birddb import ImageDataset
from models import BirdModel
from expt import Expt


if __name__ == "__main__":
    datadir = '/data1/datasets/birds'
    sname = 'nab_wood_sm'
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Env.initEnv("/home/dg/proj/birddb",datadir,device)
    ##driver = Driver(expbase,datadir,dbname,device)
    modname = "RN50v1_t1_e10"
    db = BirdDB.DBFromName(sname)
    mod = BirdModel.modelFromName(modname,db)
    print(mod)
    if not mod.loaded:
      expt = Expt.ExptFromName(modname,db)
      expt.trainModel(mod,1)
      mod.writeModelState()

    df = pd.read_csv(f"{Env.TheEnv.expdir}/{db.dbname}/{mod.modname}/trainStats.csv")
    print(df)
    df.plot()
    exit()
   