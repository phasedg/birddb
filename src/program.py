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
from imageds import ImageDataset
from models import BirdModel
from expt import Expt


if __name__ == "__main__":
    
    Env.setupEnv()
    sname = 'nab_wood_sm'
    ##driver = Driver(expbase,datadir,dbname,device)
    modname = "RN50v1_t2_e5_b32"
    db = BirdDB.DBFromName(sname)
    mod = BirdModel.modelFromName(modname,db)
    expt = Expt.ExptFromName(modname,db)
    print(mod)
    if not mod.loaded:
      expt.trainModel(mod)
      mod.writeModelState()

    df = pd.read_csv(expt.runStatFile)
    print(df)
    df.plot()
    exit()
   