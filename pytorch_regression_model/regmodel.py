import torch
import pandas as pd
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

class RegModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, dropout1=0.5, dropout2=0.5): #, hidden_size3):
        super(RegModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_size, out_features=hidden_size1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout1),
            torch.nn.Linear(in_features=hidden_size1, out_features=hidden_size2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout2),
            torch.nn.Linear(in_features=hidden_size2, out_features=1)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)
    
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def compute_l2_loss(self, w):
        return torch.square(w).sum()


class PrepareData(torch.utils.data.Dataset):

    def __init__(self, X, y, weight=None, scale_X=False):
        if not torch.is_tensor(X):
            if scale_X:
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        if weight is None:
            self.weight = torch.from_numpy(np.ones(len(y)))
        else:
            self.weight = torch.from_numpy(weight)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weight[idx]

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)

def getWeights(y_df, target_var, bins):
    bin_count = []
    for i in range(0, len(bins)):
        count = y_df.loc[(y_df[target_var] > bins[i][0]) & 
                               (y_df[target_var] <= bins[i][1])].shape[0]
        bin_count.append(count)
    
    y_list = y_df[target_var].to_list()
    y_weight = []
    for i in range(0, len(y_list)):
        ind = 0
        for j in range(0, len(bins)):
            if (y_list[i] > bins[j][0]) & (y_list[i] <= bins[j][1]):
                ind = j
        y_weight.append(np.sum(bin_count)/(len(bin_count)*bin_count[ind]))
    
    return y_weight
    

def trainRegModel(regmodel, dataLoader, epochs, loss_fn, optimizer, 
                  l1_weight=0.1, l2_weight=0.1, verbose=True):
    all_losses = []
    for e in range(epochs):
        batch_losses = []

        for batch, (X, y, weight) in enumerate(dataLoader):

            _X = X.float()
            _y = y.float()
            _weight = weight.float()

            #==========Forward pass===============

            preds = regmodel(_X)
            loss = loss_fn(preds, _y, _weight)

            #==========Regularization=============

            # Compute L1 and L2 loss component
            parameters = []
            for parameter in regmodel.parameters():
                parameters.append(parameter.view(-1))
            l1 = l1_weight * regmodel.compute_l1_loss(torch.cat(parameters))
            l2 = l2_weight * regmodel.compute_l2_loss(torch.cat(parameters))

            # Add L1 and L2 loss components
            loss += l1
            loss += l2

            #==========Backward pass==============

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.data.item())
            all_losses.append(loss.data.item())

        mbl = np.mean(np.sqrt(batch_losses)).round(3)

        if verbose and e % 5 == 0:
            print("Epoch [{}/{}], Batch loss: {}".format(e, epochs, mbl))
        
    return regmodel, all_losses

def evalValLoss(regmodel, valLoader, loss_fn):
    
    regmodel.train(False)
    all_losses = []
    
    for batch, (X, y, weight) in enumerate(valLoader):

        _X = X.float()
        _y = y.float()
        _weight = weight.float()

        preds = regmodel(_X)
        loss = loss_fn(preds, _y, _weight)

        all_losses.append(loss.data.item())
            
    return np.mean(np.sqrt(all_losses)).round(3)

def evalValCor(regmodel, X_val, y_val_list):
    
    regmodel.train(False)
    
    y_val_pred = regmodel(torch.from_numpy(X_val).float())
    y_val_pred_list = [x.data.item() for x in y_val_pred]
    
    cor = np.corrcoef(y_val_list, y_val_pred_list)[0,1]
    return cor

def evalValAUROC(regmodel, X_val, y_val_bin_list):
    
    regmodel.train(False)
    
    y_val_pred = regmodel(torch.from_numpy(X_val).float())
    y_val_pred_list = [x.data.item() for x in y_val_pred]
    
    auc = roc_auc_score(y_val_bin_list, y_val_pred_list)
    return auc
