"""
Created on 25/09/2023
TropiGAT graph functions
@author: conchaeloko
"""

# 0 - LIBRARIES
# --------------------------------------------------
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
import random
from collections import Counter
import warnings
import logging
from multiprocessing.pool import ThreadPool
warnings.filterwarnings("ignore")

# PyTorch Libraries
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import HeteroData, DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import to_hetero , HeteroConv , GATv2Conv , SAGEConv
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import LinkNeighborLoader

# Sklearn Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder , label_binarize , OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score , matthews_corrcoef

# 1 - Models
# --------------------------------------------------
# The model : TropiGAT
class TropiGAT_small_module(torch.nn.Module):
    def __init__(self,hidden_channels, heads, edge_type = ("B2", "expressed", "B1") ,dropout = 0.2, conv = GATv2Conv):
        super().__init__()
        # GATv2 module :
        self.conv = conv((-1,-1), hidden_channels, add_self_loops = False, heads = heads, dropout = dropout, shared_weights = True)
        self.hetero_conv = HeteroConv({edge_type: self.conv})
        # FNN layers : 
        self.linear_layers = nn.Sequential(nn.Linear(heads*hidden_channels, 1280),
                                           nn.BatchNorm1d(1280),
                                           nn.LeakyReLU(),
                                           torch.nn.Dropout(dropout),
                                           nn.Linear(1280, 480),
                                           nn.BatchNorm1d(480),
                                           nn.LeakyReLU(),
                                           torch.nn.Dropout(dropout),
                                           nn.Linear(480 , 1))
        
    def forward(self, graph_data):
        x_B1_dict  = self.hetero_conv(graph_data.x_dict, graph_data.edge_index_dict)
        x = self.linear_layers(x_B1_dict["B1"])
        return x.view(-1) 

class TropiGAT_big_module(torch.nn.Module):
    def __init__(self,hidden_channels, heads, edge_type = ("B2", "expressed", "B1") ,dropout = 0.2, conv = GATv2Conv):
        super().__init__()
        # GATv2 module :
        self.conv = conv((-1,-1), hidden_channels, add_self_loops = False, heads = heads, dropout = dropout, shared_weights = True)
        self.hetero_conv = HeteroConv({edge_type: self.conv})
        # FNN layers : 
        self.linear_layers = nn.Sequential(nn.Linear(heads*hidden_channels, 1280),
                                           nn.BatchNorm1d(1280),
                                           nn.LeakyReLU(),
                                           torch.nn.Dropout(dropout),
                                           nn.Linear(1280, 720),
                                           nn.BatchNorm1d(720),
                                           nn.LeakyReLU(),
                                           torch.nn.Dropout(dropout),
                                           nn.Linear(720 , 240),
                                           nn.BatchNorm1d(240),
                                           nn.LeakyReLU(),
                                           torch.nn.Dropout(dropout),
                                           nn.Linear(240, 1)
                                          )        
    def forward(self, graph_data):
        x_B1_dict = self.hetero_conv(graph_data.x_dict, graph_data.edge_index_dict)
        x = self.linear_layers(x_B1_dict["B1"])
        return x.view(-1) 


# Version of the model capturing the attention weights :
class TropiGAT_small_module_attention(torch.nn.Module):
    def __init__(self,hidden_channels, heads, edge_type = ("B2", "expressed", "B1") ,dropout = 0.2, conv = GATv2Conv):
        super().__init__()
        # GATv2 module :
        self.conv = conv((-1,-1), hidden_channels, add_self_loops = False, heads = heads, dropout = dropout, shared_weights = True, return_attention_weights = True)
        self.hetero_conv = HeteroConv({edge_type: self.conv})
        # FNN layers : 
        self.linear_layers = nn.Sequential(nn.Linear(heads*hidden_channels, 1280),
                                           nn.BatchNorm1d(1280),
                                           nn.LeakyReLU(),
                                           torch.nn.Dropout(dropout),
                                           nn.Linear(1280, 480),
                                           nn.BatchNorm1d(480),
                                           nn.LeakyReLU(),
                                           torch.nn.Dropout(dropout),
                                           nn.Linear(480 , 1))
        
    def forward(self, graph_data):
        x_B1_dict, weights  = self.conv((graph_data.x_dict["B2"], graph_data.x_dict["B1"]), graph_data.edge_index_dict[("B2", "expressed", "B1")], return_attention_weights=True)
        x = self.linear_layers(x_B1_dict)
        return x.view(-1), weights 

class TropiGAT_big_module_attention(torch.nn.Module):
    def __init__(self,hidden_channels, heads, edge_type = ("B2", "expressed", "B1") ,dropout = 0.2, conv = GATv2Conv):
        super().__init__()
        # GATv2 module :
        self.conv = conv((-1,-1), hidden_channels, add_self_loops = False, heads = heads, dropout = dropout, shared_weights = True, return_attention_weights = True)
        self.hetero_conv = HeteroConv({edge_type: self.conv})
        # FNN layers : 
        self.linear_layers = nn.Sequential(nn.Linear(heads*hidden_channels, 1280),
                                           nn.BatchNorm1d(1280),
                                           nn.LeakyReLU(),
                                           torch.nn.Dropout(dropout),
                                           nn.Linear(1280, 720),
                                           nn.BatchNorm1d(720),
                                           nn.LeakyReLU(),
                                           torch.nn.Dropout(dropout),
                                           nn.Linear(720 , 240),
                                           nn.BatchNorm1d(240),
                                           nn.LeakyReLU(),
                                           torch.nn.Dropout(dropout),
                                           nn.Linear(240, 1)
                                          )        
    def forward(self, graph_data):
        x_B1_dict , weights = self.conv((graph_data.x_dict["B2"], graph_data.x_dict["B1"]),graph_data.edge_index_dict[("B2", "expressed", "B1")] , return_attention_weights=True)
        x = self.linear_layers(x_B1_dict)
        return x.view(-1) , weights



# GrapheSage version of the model : 
class TropiGAT_small_sage_module(torch.nn.Module):
    def __init__(self,hidden_channels, edge_type = ("B2", "expressed", "B1") ,dropout = 0.2, conv = SAGEConv):
        super().__init__()
        # GATv2 module :
        self.conv = conv((-1,-1), hidden_channels)
        self.hetero_conv = HeteroConv({edge_type: self.conv})
        # FNN layers : 
        self.linear_layers = nn.Sequential(nn.Linear(hidden_channels, 1280),
                                           nn.BatchNorm1d(1280),
                                           nn.LeakyReLU(),
                                           torch.nn.Dropout(dropout),
                                           nn.Linear(1280, 480),
                                           nn.BatchNorm1d(480),
                                           nn.LeakyReLU(),
                                           torch.nn.Dropout(dropout),
                                           nn.Linear(480 , 1))
        
    def forward(self, graph_data):
        x_B1_dict  = self.hetero_conv(graph_data.x_dict, graph_data.edge_index_dict)
        x = self.linear_layers(x_B1_dict["B1"])
        return x.view(-1) 

class TropiGAT_big_sage_module(torch.nn.Module):
    def __init__(self,hidden_channels, edge_type = ("B2", "expressed", "B1") ,dropout = 0.2, conv = SAGEConv):
        super().__init__()
        # GATv2 module :
        self.conv = conv((-1,-1), hidden_channels)
        self.hetero_conv = HeteroConv({edge_type: self.conv})
        # FNN layers : 
        self.linear_layers = nn.Sequential(nn.Linear(hidden_channels, 1280),
                                           nn.BatchNorm1d(1280),
                                           nn.LeakyReLU(),
                                           torch.nn.Dropout(dropout),
                                           nn.Linear(1280, 720),
                                           nn.BatchNorm1d(720),
                                           nn.LeakyReLU(),
                                           torch.nn.Dropout(dropout),
                                           nn.Linear(720 , 240),
                                           nn.BatchNorm1d(240),
                                           nn.LeakyReLU(),
                                           torch.nn.Dropout(dropout),
                                           nn.Linear(240, 1)
                                          )        
    def forward(self, graph_data):
        x_B1_dict = self.hetero_conv(graph_data.x_dict, graph_data.edge_index_dict)
        x = self.linear_layers(x_B1_dict["B1"])
        return x.view(-1) 

# 2 - Training
# --------------------------------------------------
def train(model, graph):
    model.train()
    optimizer.zero_grad()
    out_train = model(graph)
    loss = criterion(out_train[graph["B1"].train_mask], graph["B1"].y[graph["B1"].train_mask])
    loss.backward()
    optimizer.step()
    return loss
	

@torch.no_grad()
def evaluate(model, graph,criterion, mask):
    model.eval()
    out_eval  = model(graph)
    logging.info(mask.shape)
    pred = out_eval[mask]
    pred = torch.sigmoid(out_eval).round()
    labels = graph["B1"].y[mask]
    val_loss = criterion(out_eval[mask], graph["B1"].y[mask].float())
    # Calculate the metrics
    f1 = f1_score(labels, pred, average='binary')
    precision = precision_score(labels, pred, average='binary')
    recall = recall_score(labels, pred, average='binary')
    mcc = matthews_corrcoef(labels, pred)
    accuracy = accuracy_score(labels, pred)
    auc = roc_auc_score(labels, out_eval[mask])
    return val_loss.item(), (f1, precision, recall, mcc, accuracy, auc) 

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, path, rounds = 5, metric='loss', patience=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            path (str): Path for the checkpoint to be saved to.
            metric (str): Metric to monitor for early stopping.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None  # add this line to keep the best model
        self.metric = metric
		#self.rounds = rounds

    def __call__(self, val_metric, model):
        if self.metric == 'loss':
            score = -val_metric
        else:
            score = val_metric
        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict()  
        elif score <= self.best_score:
            self.counter += self.rounds
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = model.state_dict()  
            self.counter = 0
    def save_checkpoint(self, model):
        '''Saves model when early stopping is triggered.'''
        torch.save(model, self.path)



		
if __name__ == "__main__":
    main()
