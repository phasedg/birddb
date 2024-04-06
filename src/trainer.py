import time
import os
import tempfile
import torch


class Trainer:
    def __init__(self,model, device, traindl, valdl, criterion, optimizer, scheduler, writer,dotfreq=0, chkpath='check.pt'):
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
        self.chkpath = chkpath

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
            running_correct += torch.sum(preds == labels)                
                
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
                running_correct += torch.sum(preds == labels)                
        last_loss = running_loss / len(self.valdl.dataset) # loss per batch
        last_acc = running_correct / len(self.valdl.dataset) # loss per batch

        return last_loss, last_acc

    def train(self,epochs):
#        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')            
        losses = []
        accs = []
        self.model = self.model.to(self.device)
        for i in range(self.epoch,epochs):
            print(f'Epoch {i}: ',end='')
            loss,acc = self.train_one_epoch(i)
            if self.scheduler is not None:
               self.scheduler.step()
            if ((i+1)%5) == 0:
                if self.chkpath is not None:
                    self.checkpoint(i)
                if (self.valdl is not None):
                    tloss,tacc = self.val_one_epoch(i)                
                    print(f' Train: Loss:{loss:.3f} Acc:{acc:.3f} Test: Loss:{tloss:.3f} Acc:{tacc:.3f}')
                else:
                    print(f' Train: Loss:{loss:.3f} Acc:{acc:.3f}')
            else:
                print(f' Train: Loss:{loss:.3f} Acc:{acc:.3f}')
            if self.writer is not None:
                self.writer.add_scalar('Train/Loss:', loss,i)
                self.writer.add_scalar('Train/Acc:', acc,i)
            losses.append(loss)
            accs.append(acc)
            
        return losses,accs

    def checkpoint(self,epoch):
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'opt_state': self.optimizer.state_dict(),
            'sched_state': self.scheduler.state_dict()
            }, self.chkpath)

    def restore(self):
        cpt = torch.load(self.chkpath,map_location='cpu')
        self.model.load_state_dict(cpt['model_state'])
        self.optimizer.load_state_dict(cpt['opt_state'])
        self.scheduler.load_state_dict(cpt['sched_state'])
        self.epoch = cpt['epoch']
        print(f'return to epoch {self.epoch}')

        

