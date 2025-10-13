import time
import os
import tempfile
import torch
from datetime import datetime


class Trainer:
    def __init__(self,model, device, traindl, valdl, criterion, optimizer, scheduler, writer,dotfreq=0):
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.writer = writer
        self.traindl = traindl
        self.valdl = valdl
        self.optimizer = optimizer
        self.device = device
        self.dotfreq = dotfreq
        self.epoch = 0
        self.losses = []
        self.accs = []


        #training loop from pytorchs docs
    def train_one_epoch(self,epoch_index):
     # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        running_loss = 0.
        last_loss = 0.
        running_correct = 0.
        last_acc = 0.
        
        self.model.train()  # put in training mode
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        fortime = 0
        backtime = 0
        opttime  = 0
        gtime=0
        for i, data in enumerate(self.traindl):
          
            # Every data instance is an input + label pair
          inputs, labels = data
          inputs = inputs.to(self.device)
          labels = labels.to(self.device)
          if self.dotfreq == 0 or (i+1)%self.dotfreq == 0:
              print('.',end='',flush=True)
          
          gstart = datetime.now()
          with torch.set_grad_enabled(True):
              
              # Zero your gradients for every batch!
              start = datetime.now()
              self.optimizer.zero_grad()
              
              # Make predictions for this batch
              outputs = self.model.forward(inputs)
              _, preds = torch.max(outputs, 1)
             # torch.cuda.synchronize()   timing bogus w/o sync
              fortime += (datetime.now()-start).total_seconds()
              ###########################################################
              # Compute the loss and its gradients
              start = datetime.now()
              loss = self.criterion(outputs, labels)
              
              loss.backward()
              backtime += (datetime.now()-start).total_seconds()
              ###########################################################
                # Adjust learning weights
              
          self.optimizer.step()
          
          ostart = datetime.now()
            # Gather data and report
          running_loss += loss.item() * inputs.size(0)
          running_correct += torch.sum(preds == labels).item()
          gtime += (datetime.now()-gstart).total_seconds()       
          opttime += (datetime.now()-ostart).total_seconds()
        last_loss = running_loss / len(self.traindl.dataset) # loss per batch
        last_acc = running_correct / len(self.traindl.dataset) # loss per batch
                
      #print(prof)
        return last_loss, last_acc, gtime

    def val_one_epoch(self,epoch_index):
        running_loss = 0.
        last_loss = 0.
        running_correct = 0.
        last_acc = 0.
        gtime = 0
        self.model.eval()  # put in training mode
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        gstart = datetime.now()
        for i, data in enumerate(self.valdl):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            with torch.no_grad():
                # Make predictions for this batch
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                # Compute the loss and its gradients
                loss = self.criterion(outputs, labels)
                
                # Gather data and report
                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels).item()
                gtime += (datetime.now()-gstart).total_seconds()         
        last_loss = running_loss / len(self.valdl.dataset) # loss per batch
        last_acc = running_correct / len(self.valdl.dataset) # loss per batch

        return last_loss, last_acc, gtime

    def train(self,epochs,callback=None):
        print(f"training on {self.device}")
#        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')            
        self.model = self.model.to(self.device)
        tloss = 0
        tacc = 0
        stats = []
        start = self.epoch
        for i in range(start,epochs):
            start = datetime.now()
            print(f'Epoch {i}: ',end='')
            # do the work
            loss,acc,gt = self.train_one_epoch(i)

            if self.scheduler is not None:
               self.scheduler.step()
            etime = (datetime.now()-start).total_seconds()
            
            self.accs.append(acc)
            self.epoch = i+1  # epoch to restart at
            if callback is not None:
                callback(i)
            if (self.valdl is not None):
                tloss,tacc,gt = self.val_one_epoch(i)                
                print(f' Train: Loss:{loss:.3f} Acc:{acc:.3f} Val: Loss:{tloss:.3f} Acc:{tacc:.3f}, ETime {etime:.3f}, GTime {gt:.3f}')
            else:
                print(f' Train: Loss:{loss:.3f} Acc:{acc:.3f}, ETime {etime:.3f} GTime {gt:.3f}')
            stats.append([loss,acc,tloss,tacc,etime,gt])
            if self.writer is not None:
                self.writer.add_scalar('Train/Loss:', loss,i)
                self.writer.add_scalar('Train/Acc:', acc,i)
        return stats
        
    '''
      Restore state from checkpoint file. Note: checkpoitn stores state not structure so trainer must be rebuilt with __init__ before restoring
    '''
    def restore(self,chkpath):
        cpt = torch.load(chkpath,map_location=self.device)
        self.model.load_state_dict(cpt['model_state'])
        self.model.to(self.device)
        self.optimizer.load_state_dict(cpt['opt_state'])
        self.scheduler.load_state_dict(cpt['sched_state'])
        self.epoch = cpt['epoch']
        if 'losses' in cpt:
            self.losses = cpt['losses']
            self.accs = cpt['accs']
        print(f'restore to epoch {self.epoch}')

    '''
     Stores trainer state (not structure)
    '''
    def checkpoint(self,chkpath):
        torch.save({
            'epoch': self.epoch,
            'model_state': self.model.state_dict(),
            'opt_state': self.optimizer.state_dict(),
            'sched_state': self.scheduler.state_dict(),
            'losses': self.losses,
            'accs': self.accs
        }, chkpath)
