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
from expt2 import Expt2
from testRun import TestRun


if __name__ == "__main__":
    
    Env.setupEnv()
    moddbsname = 'cub_bb4'
    testdbsname = 'cub_bb4'
    
    ##driver = Driver(expbase,datadir,dbname,device)
    modname = "RN50v2_u1_e20_b32_l05_L4"
    db = BirdDB.DBFromName(moddbsname)
    
    mod = BirdModel.modelFromName(modname,db)
    expt = Expt2.ExptFromName(modname,db)

    print(mod)
    if not mod.loaded and modname.endswith("L4"):
      expt.trainModel(mod)
      
      exit()

    expt.model = mod
    
    rundb = BirdDB.DBFromName(testdbsname)
    tdb = rundb.getTestDB()
    tr = expt.runOnDB(tdb)
    print(tr.top1Acc())
    print(tr.topNAcc(2))
    print(tr.topNAcc(5))
    print(tr.top1Acc(0.9))
    
    exit()

    tr = expt.runOnDB(rundb.getTestDB())
    
    if testdbsname == moddbsname:
      
      print(tr.top1Acc())
      print(tr.top1Acc(0.6))
      print(tr.top1Acc(0.9))
     # print(tr.classAcc()[0])
     # print(tr.classAcc()[1])
     # for i in ['10','11','12']:
     #   tr.visualizeErrors(i)
        
    

          

          

    exit()
   