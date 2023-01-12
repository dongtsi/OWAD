import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import myutils as utils
Params = utils.get_params('DeepLog')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_keys, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)
        # self.sm = nn.Softmax()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        # out = self.sm(out)
        return out

def train(X_train, epoches = Params['num_epochs']):
    # X_train = {input:[[]], output:[]}
    hidden_size = Params['hidden_size']
    num_layers = Params['num_layers']
    num_classes = Params['num_classes']
    X_input, X_output = X_train['input'], X_train['output']
    dataset = TensorDataset(torch.tensor(X_input, dtype=torch.long), torch.tensor(X_output))
    dataloader = DataLoader(dataset, batch_size=Params['batch_size'], shuffle=True, pin_memory=True)
    
    model = Model( hidden_size, num_layers, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=Params['lr'])
    
    total_step = len(dataloader)
    for epoch in range(epoches):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            ## Forward pass
            seq = seq.clone().detach().view(-1, Params['window_size']).to(device)
            seq = F.one_hot(seq,num_classes=num_classes).float()
            # print(seq.shape,seq)
            output = model(seq)
            loss = criterion(output, label.to(device))
            ## Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, epoches, train_loss / total_step))
    print('Finished Training')

    return model

@torch.no_grad()
def test(model, X_test, num_classes=Params['num_classes']):
    model.eval()
    Labels = []
    Preds = []
    
    X_input, X_output = X_test['input'], X_test['output'] 

    
    with torch.no_grad():

        ## sample-wise fashion
        # cnt = 0
        # for seq, label in zip(X_input, X_output):
        #     seq = torch.tensor(seq, dtype=torch.long).view(-1, Params['window_size']).to(device)
        #     seq = F.one_hot(seq,num_classes=Params['num_classes']).float()
        #     label = torch.tensor(label).view(-1).to(device)
        #     output = model(seq)
        #     predicted = torch.argsort(output, 1)[0][-Params['num_candidates']:]
        #     Preds.append(output[0, label].cpu().item())
        #     if label in predicted:
        #         Labels.append(0)
        #     else:
        #         Labels.append(1)
        #     cnt += 1
        #     if cnt%1000 == 0:
        #         print('Testing: %d/%d'%(cnt, len(X_input)))

        ## all-in-one fashion
        # seq = torch.tensor(X_input, dtype=torch.long).view(-1, Params['window_size']).to(device)
        # seq = F.one_hot(seq,num_classes=Params['num_classes']).float()
        # label = torch.tensor(X_output).view(-1).to(device)
        # output = model(seq)
        # predicted = torch.argsort(output, 1)[:, -Params['num_candidates']:]
        # for i in range(len(label)):
        #     if label[i] in predicted[i]:
        #         Labels.append(0)
        #     else:
        #         Labels.append(1)
        # Preds = output.cpu().detach().numpy()[np.arange(len(X_output)),X_output]

        ## batch fashion considering the limit of GPU memory
        test_batch = Params['test_batch'] # depends on the available GPU memory (~5GB for <test_batch>:20000, <num_classes>:1600, <window_size>:10)
        test_steps = len(X_input)//test_batch
        if len(X_input) % test_batch != 0:
            test_steps += 1
        for i in range(test_steps):
            seq = torch.tensor(X_input[test_batch*i:test_batch*(i+1)], dtype=torch.long).view(-1, Params['window_size']).to(device)
            seq = F.one_hot(seq,num_classes=num_classes).float()
            label = torch.tensor(X_output[test_batch*i:test_batch*(i+1)]).view(-1).to(device)
            output = model(seq)
            predicted = torch.argsort(output, 1)[:, -Params['num_candidates']:]
            for j in range(len(label)):
                if label[j] in predicted[j]:
                    Labels.append(0)
                else:
                    Labels.append(1)
            preds = output.cpu().detach().numpy()[np.arange(len(X_output[test_batch*i:test_batch*(i+1)])),X_output[test_batch*i:test_batch*(i+1)]]
            if i == 0:
                Preds = preds
            else:
                Preds = np.concatenate((Preds,preds))
            if Params['test_info']:
                print('deeplog model testing: %d/%d'%(i,test_steps))
        
    return Labels, Preds
