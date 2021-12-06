import torch
import time

from utils import get_batch

class Trainer:

    def __init__(self, train_loader, test_loader, model, loss, optimizer) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.interval = 200


    def train(self, train_losses, epoch, batch_size, clip) -> list:  
        # Initialization of RNN hidden, and cell states.
        states = self.model.init_hidden(batch_size) 

        for batch_num, batch in enumerate(self.train_loader): # loop over the data, and jump with step = bptt.
            # get the labels
            source, target, source_lengths = get_batch(batch)
            source = source.to(self.device)
            target = target.to(self.device)


            pred, states = self.model(source,source_lengths, states)

            # detach hidden states
            states = states[0].detach(), states[1].detach()

            # compute the loss
            pred = pred.view(-1,pred.size(2)).contiguous()  # reshape the input (pred, target) from (batch_size, seq_leng, vocab_size)  
            target = target.view(-1).contiguous()           # to (batch_size*seq_leng, vocab_size)
            mloss = self.loss(pred, target)

            train_losses.append(mloss.item())

            mloss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            self.optimizer.step()

            self.optimizer.zero_grad()


            if batch_num % self.interval == 0 and batch_num > 0:
  
                print('| epoch {:3d} | loss {:5.6f} '.format(epoch, mloss.item()))
            

        return train_losses

    def test(self, test_losses, epoch, batch_size) -> list:

        with torch.no_grad():

            states = self.model.init_hidden(batch_size) 

            for batch_num, batch in enumerate(self.test_loader): # loop over the data, and jump with step = bptt.
                # get the labels
                source, target, source_lengths = get_batch(batch)
                source = source.to(self.device)
                target = target.to(self.device)


            pred, states = self.model(source,source_lengths, states)

            # detach hidden states
            states = states[0].detach(), states[1].detach()

            # compute the loss
            pred = pred.view(-1,pred.size(2)).contiguous()  # reshape the input (pred, target) from (batch_size, seq_leng, vocab_size)  
            target = target.view(-1).contiguous()           # to (batch_size*seq_leng, vocab_size)
            mloss = self.loss(pred, target)

            test_losses.append(mloss)


            if batch_num % self.interval == 0 and batch_num > 0:
  
                print('| epoch {:3d} | loss {:5.6f} '.format(epoch, mloss.item()))

                # Statistics.
                # if batch_num % 20 ==0:
                #   print('| epoch {:3d} | loss {:5.6f} '.format(  epoch, mloss.item()))

            return test_losses