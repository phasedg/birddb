import time
import os
import tempfile
import torch


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
        running_loss = 0.
        last_loss = 0.
        running_correct = 0.
        last_acc = 0.
        
        self.model.train()  # put in training mode
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.traindl):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            if self.dotfreq == 0 or (i+1)%self.dotfreq == 0:
                print('.',end='',flush=True)
            with torch.set_grad_enabled(True):
                # Zero your gradients for every batch!
                self.optimizer.zero_grad()
                
                # Make predictions for this batch
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                # Compute the loss and its gradients
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Adjust learning weights
            self.optimizer.step()
                
                # Gather data and report
            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(preds == labels).item()        
                
        last_loss = running_loss / len(self.traindl.dataset) # loss per batch
        last_acc = running_correct / len(self.traindl.dataset) # loss per batch
                

        return last_loss, last_acc

    def val_one_epoch(self,epoch_index):
        running_loss = 0.
        last_loss = 0.
        running_correct = 0.
        last_acc = 0.
        self.model.eval()  # put in training mode
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
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
                running_loss += loss.item()
                running_correct += torch.sum(preds == labels).item()          
        last_loss = running_loss / len(self.valdl.dataset) # loss per batch
        last_acc = running_correct / len(self.valdl.dataset) # loss per batch

        return last_loss, last_acc

    def train(self,epochs,callback=None):
#        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')            
        self.model = self.model.to(self.device)
        start = self.epoch
        for i in range(start,epochs):
            print(f'Epoch {i}: ',end='')
            loss,acc = self.train_one_epoch(i)
            if self.scheduler is not None:
               self.scheduler.step()
            self.losses.append(loss)
            self.accs.append(acc)
            self.epoch = i+1  # epoch to restart at
            if callback is not None:
                callback(i)
            if (self.valdl is not None) and ((i+1)%5) == 0:
                tloss,tacc = self.val_one_epoch(i)                
                print(f' Train: Loss:{loss:.3f} Acc:{acc:.3f} Test: Loss:{tloss:.3f} Acc:{tacc:.3f}')
            else:
                print(f' Train: Loss:{loss:.3f} Acc:{acc:.3f}')
            if self.writer is not None:
                self.writer.add_scalar('Train/Loss:', loss,i)
                self.writer.add_scalar('Train/Acc:', acc,i)
        return self.losses,self.accs

        
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
