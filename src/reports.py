import time
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
sys.path.append('../src')
from env import Env
from birddb import BirdDB
from imageds import ImageDataset
from models import BirdModel
from expt import Expt
from testRun import TestRun
Env.setupEnv()

class Report:

  def __init__(self):

    self.html = ""
    self.openTag("html")
    self.openTag("head")
    self.closeTag("head")
    self.openTag("body")

  def append(self,tags):
     self.html += tags

  def openTag(self,tag,cls=None,id=None):
     ct = f' class="{cls} "' if cls is not None else ""
     ci = f' id="{id} "' if id is not None else ""
     s = f"<{tag}{ct}{ci}>\n"
     self.html += s
  
  def closeTag(self,tag):
     self.html += f"</{tag}>\n"

  def closeHtml(self):
     self.closeTag("body")
     self.closeTag("html")

  def writeToFile(self,file):
    with open(file,'w') as f:
      f.write(self.html)
     



  def topNbars(self,dbname,modlist,labs):
      db = BirdDB.DBFromName(dbname)
    
      top1 = []
      top2 = []
      top5 = []
      for modname in modlist:
        print(modname,os.getcwd())
        tr = TestRun(f'./trainrun/{dbname}/{modname}',modname,db,f'{dbname}_tst')
      # print(f't{i}, {tr.top1Acc()[0]:0.2f},{tr.top1Acc(.5)[0]:0.2f},{tr.top1Acc(.9)[0]:0.2f},{tr.top1Acc(.9)[1]}, {tr.topNAcc(2)[0]:0.2f},{tr.topNAcc(5)[0]:0.2f}')
        
        top1.append(tr.top1Acc()[0])
        top2.append(tr.topNAcc(2)[0])
        top5.append(tr.topNAcc(5)[0])
      fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(12,3))
      axes[0].set_title("Top1")
      axes[0].set_ylim(0,1.0)
      axes[1].set_title("Top2")
      axes[1].set_ylim(0,1.0)
      axes[2].set_title("Top5")
      axes[2].set_ylim(0,1.0)
      axes[0].bar(labs,top1)
      axes[1].bar(labs,top2)
      axes[2].bar(labs,top5)
      df = pd.DataFrame({'Lab':labs,'Top1':top1,'Top2':top2,'Top5':top5})
      return df,axes,fig
  
  def plotTrainStats(self,files,labs):

    dfs = {}
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(12,6))
    axes[0,0].set_title("Vacc")
    axes[0,0].set_ylim(0,1.0)
    axes[0,1].set_title("Acc")
    axes[0,1].set_ylim(0,1.0)
    axes[1,0].set_title("Loss")
    axes[1,1].set_title("ETime")
    for i,runfile in enumerate(files):
      df = pd.read_csv(runfile)
      dfs[i] = df
    
    vacc = pd.DataFrame()
    for i,l in enumerate(labs):
        vacc[l] = dfs[i]['Vacc']
    vacc.plot(ax=axes[0,0])
    vacc = pd.DataFrame()
    for i,l in enumerate(labs):
        vacc[l] = dfs[i]['Acc']
    vacc.plot(ax=axes[0,1])
    for i,l in enumerate(labs):
        vacc[l] = dfs[i]['Loss']
    vacc.plot(ax=axes[1,0])
    for i,l in enumerate(labs):
        vacc[l] = dfs[i]['ETime']
    vacc.plot(ax=axes[1,1])
    return fig,axes

  def imgTagForFig(self,fig):
    my_stringIObytes = io.BytesIO()
    fig.savefig(my_stringIObytes, format='png')
    my_stringIObytes.seek(0)
    my_base64 = base64.b64encode(my_stringIObytes.read()).decode()
    img = f'<img src="data:image/png;base64,{my_base64}">'
    return img

class Report_Expt1(Report):
   
  def __init(self):
    super().__init__()

  def makeReport(self):
    modlist = [f'RN50v1_t{i}_e60_b32_L4' for i in [2,3,4,5,6,7,8]]
    labs = [f't{i}' for i in [2,3,4,5,6,7,8]]
    topn,axes,fig = self.topNbars('cub_sm',modlist,labs)
    img = self.imgTagForFig(fig)
    self.append(topn.to_html())
    self.append(img)
    
    files = [f'./trainrun/cub_sm/expt1/RN50v1_t{i}_e60_b32_L4/trainStats.csv' for i in [2,3,4,5,6,7,8]]
    labs = [f't{i}' for i in [2,3,4,5,6,7,8]]
    fig, axes = self.plotTrainStats(files,labs)
    img = self.imgTagForFig(fig)
    self.append(img)



    files = [f'./trainrun/cub_sm/expt1/RN50v1_t{i}_e40_b32_L4/trainStats.csv' for i in [7,8]]
    labs = [f't{i}_40' for i in [7,8]]
    files += [f'./trainrun/cub_sm/expt1/RN50v1_t{i}_e60_b32_L4/trainStats.csv' for i in [7,8]]
    labs += [f't{i}_60' for i in [7,8]]
    fig, axes = self.plotTrainStats(files,labs)
    img = self.imgTagForFig(fig)
    self.append(img)

    modlist = [f'RN50v1_{i}_b32_L4' for i in ['t7_e40','t7_e60','t8_e40']]
    labs = ['t7_e40','t7_e60','t8_e40']

    topn,axes,fig = self.topNbars('cub_sm',modlist,labs)
    img = self.imgTagForFig(fig)
    self.append(topn.to_html())
    self.append(img)

    files = [f'./trainrun/nab_sm/expt1/RN50v1_{i}_b32_L4/trainStats.csv' for i in ['t7_e40','t7_e60','t8_e40']]
    labs = ['t7_e40','t7_e60','t8_e40']

    fig, axes = self.plotTrainStats(files,labs)
    img = self.imgTagForFig(fig)
    self.append(img)

    modlist = ['RX50v2_t9_e20_b32_l01_L4','RX50v2_t9_e20_b32_l05_L4','RN50v2_t9_e20_b32_l01_L4','RN50v2_t9_e20_b32_l05_L4']
    labs = ['RX_l01','RX_l05','RN_l01','RN_l05']

    topn,axes,fig = self.topNbars('cub_sm',modlist,labs)
    img = self.imgTagForFig(fig)
    self.append(topn.to_html())
    self.append(img)


class Report_Expt2(Report):
   
  def __init(self):
    super().__init__()

  def makeReport(self):
    modlist = ['RN50v2_u1_e20_b32_l05_L4','RN50v2_u1_e20_b32_l05_d49_L4','RN50v2_u1_e30_b32_l01_L4','RN50v2_u1_e30_b32_l02_L4']
    labs = [f"{mod.replace('RN50v2_u1_','').replace('_L4','').replace('_b32','')}" for mod in modlist]
    topn,axes,fig = self.topNbars('cub_sm',modlist,labs)
    img = self.imgTagForFig(fig)
    self.append(topn.to_html())
    self.append(img)
    
    files = [f'./trainrun/cub_sm/{mod}/trainStats.csv' for mod in modlist]
   
    fig, axes = self.plotTrainStats(files,labs)
    img = self.imgTagForFig(fig)
    self.append(img)

    modlist = ['RN50v2_u1_e20_b32_l05_d49_L4','RN50v2_u1_e20_b32_l05_d49_L4']
    labs = [f"{mod.replace('RN50v2_u1_','').replace('_L4','').replace('_b32','')}" for mod in modlist]
    topn,axes,fig = self.topNbars('cub_sm',modlist,labs)
    img = self.imgTagForFig(fig)
    self.append(topn.to_html())
    self.append(img)
    
    files = [f'./trainrun/cub_sm/{mod}/trainStats.csv' for mod in modlist]
   
    fig, axes = self.plotTrainStats(files,labs)
    img = self.imgTagForFig(fig)
    self.append(img)


    modlist = ['RN50v2_u1_e30_b32_l02_L4','RX50v2_u1_e30_b32_l02_L4']
    labs = [f"{mod.replace('RN50v2_u1_','').replace('_L4','').replace('_b32','')}" for mod in modlist]
    topn,axes,fig = self.topNbars('cub_sm',modlist,labs)
    img = self.imgTagForFig(fig)
    self.append(topn.to_html())
    self.append(img)
    
    files = [f'./trainrun/cub_sm/{mod}/trainStats.csv' for mod in modlist]
   
    fig, axes = self.plotTrainStats(files,labs)
    img = self.imgTagForFig(fig)
    self.append(img)

    modlist = ['RN50v2_u1_e20_b32_l05_L4','RN101v2_u1_e20_b32_l05_L4','RX101v2_u1_e20_b32_l05_L4']
    labs = [f"{mod.replace('RN50v2_u1_','').replace('_L4','').replace('_b32','')}" for mod in modlist]
    topn,axes,fig = self.topNbars('cub_sm',modlist,labs)
    img = self.imgTagForFig(fig)
    self.append(topn.to_html())
    self.append(img)
    
    files = [f'./trainrun/cub_sm/{mod}/trainStats.csv' for mod in modlist]
   
    fig, axes = self.plotTrainStats(files,labs)
    img = self.imgTagForFig(fig)
    self.append(img)

    modlist = ['RN101v2_u1_e20_b32_l05_L4','RX101v2_u1_e20_b32_l05_L4','RN101v2_u1_e10_b32_l05_L4','RX101v2_u1_e10_b32_l05_L4']
    labs = [f"{mod.replace('RN50v2_u1_','').replace('_L4','').replace('_b32','')}" for mod in modlist]
    topn,axes,fig = self.topNbars('cub_sm',modlist,labs)
    img = self.imgTagForFig(fig)
    self.append(topn.to_html())
    self.append(img)
    
    files = [f'./trainrun/cub_sm/{mod}/trainStats.csv' for mod in modlist]
   
    fig, axes = self.plotTrainStats(files,labs)
    img = self.imgTagForFig(fig)
    self.append(img)


    



    
if __name__ == '__main__':
   rpt = Report_Expt2()
   rpt.makeReport()
   rpt.writeToFile("Report_Expt2.html")
