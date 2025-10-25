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
    moddbsname = 'cub_sm'
    testdbsname = 'cub_sm'
    ##driver = Driver(expbase,datadir,dbname,device)
    modname = "RN50v1_t8_e40_b32_L4"
    db = BirdDB.DBFromName(moddbsname)
    for i in range(2,8):
      modname = f"RN50v1_t{i}_e60_b32_L4"
      tr = TestRun(f'./trainrun/{moddbsname}/{modname}',modname,db,f'{testdbsname}_tst')
      print(f't{i}, {tr.top1Acc()[0]:0.2f},{tr.top1Acc(.5)[0]:0.2f},{tr.top1Acc(.9)[0]:0.2f},{tr.top1Acc(.9)[1]}, {tr.topNAcc(2)[0]:0.2f},{tr.topNAcc(5)[0]:0.2f}')
    
    
    exit()
    
    mod = BirdModel.modelFromName(modname,db)
    expt = Expt.ExptFromName(modname,db)

    print(mod)
    if not mod.loaded and modname.endswith("py"):
      expt.trainModel(mod)
      
      exit()

    expt.model = mod
    
    rundb = BirdDB.DBFromName(testdbsname)
    tr = expt.runOnDB(rundb.getTestDB())
    
    if testdbsname == moddbsname:
      
      print(tr.top1Acc())
      print(tr.top1Acc(0.6))
      print(tr.top1Acc(0.9))
        
    

          

          

    exit()
   