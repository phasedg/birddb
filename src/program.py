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
from testRun import TestRun


if __name__ == "__main__":
    
    Env.setupEnv()
    moddbsname = 'nab_sm'
    ##driver = Driver(expbase,datadir,dbname,device)
    modname = "RN50v1_t3_e60_b32_L4"
    db = BirdDB.DBFromName(moddbsname)
    mod = BirdModel.modelFromName(modname,db)
    expt = Expt.ExptFromName(modname,db)

    print(mod)
    if not mod.loaded and modname.endswith("L4"):
      expt.trainModel(mod)
      mod.writeModelState()
      exit()

    expt.model = mod
    testdbsname = 'nab_sm'
    rundb = BirdDB.DBFromName(testdbsname)
    tr = expt.runOnDB(rundb.getTestDB())
    
    if testdbsname == moddbsname:
      
      print(tr.top1Acc())
      print(tr.top1Acc(0.6))
      print(tr.top1Acc(0.9))
        
    

          

          

    exit()
   