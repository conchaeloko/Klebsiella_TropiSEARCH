"""
Created on 04/09/2023
TropiGAT functions
@author: conchaeloko
"""
# 0 - LIBRARIES
# --------------------------------------------------
from torch_geometric.data import HeteroData, DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import to_hetero , HeteroConv , GATv2Conv
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import LinkNeighborLoader

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder , label_binarize , OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score , matthews_corrcoef

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
import random
from collections import Counter
import warnings
import logging
import pprint
from multiprocessing.pool import ThreadPool

path_work = "/media/concha-eloko/Linux/PPT_clean"
path_ensemble = f"{path_work}/ficheros_28032023/ensemble_1908"

# 1 - FUNCTIONS
# --------------------------------------------------
def graph_single_Dpo_pred(df_embeddings , path_work_f = "/media/concha-eloko/Linux/PPT_clean" , path_ensemble_f = f"/media/concha-eloko/Linux/PPT_clean/ficheros_28032023/ensemble_1908") : 
	""" 
	Inputs : 
	Outputs : 

	Function that 
	"""
	#path_work = "/media/concha-eloko/Linux/PPT_clean"
	#path_ensemble = f"{path_work}/ficheros_28032023/ensemble_1908"
    # Load the template data :
    pred_data_single = torch.load(f'{path_work_f}/template_graph.KLtypes.pt')
    str_keys_dico_kltype = json.load(open(f"{path_work_f}/tensorToKLtypes.json"))
    dico_kltype = {tuple(map(int, key.strip('()').split(','))): value for key, value in str_keys_dico_kltype.items()}    
    # Defining the nodes :
    n_dpos = len(df_embeddings)
    pred_data_single["B1"].x = torch.empty((n_dpos, 0))
    pred_data_single["B2"].x = torch.tensor(df_embeddings.iloc[:, :1280].values , dtype=torch.float)
    # Defining the edge_file :
    edge_index_B2_B1 = torch.tensor([[i , i] for i in range(n_dpos)] , dtype=torch.long)
    pred_data_single['B2', 'expressed', 'B1'].edge_index = edge_index_B2_B1.t().contiguous()
    edge_index_B1_A = torch.tensor([[i,j] for i in range(n_dpos) for j in range(len(pred_data_single["A"].x))] , dtype=torch.long)
    pred_data_single['B1', 'infects', 'A'].edge_label_index = edge_index_B1_A.t().contiguous()
    return pred_data_single

def get_nodes_id_single(B1A_index_file) :
    B1A_index_file = B1A_index_file.numpy()
    B1A_index_file = tuple(zip(B1A_index_file[0],B1A_index_file[1]))
    id_file = [(dpo_embeddings.index[tup[0]] , dico_kltype[tuple(pred_data_single["A"]["x"][tup[1]].numpy())]) for tup in B1A_index_file]
    return id_file

@torch.no_grad()
def make_predictions(model, data):
	""" 
	Inputs : 
	Outputs : 

	Function that  
	"""
    model.eval() 
    output = model(data)
    probabilities = torch.sigmoid(output)  # Convert output to probabilities
    predictions = probabilities.round()  # Convert probabilities to class labels
    return predictions, probabilities

def run_predictions(graph, ratios = [1,2,3,4,6,7], path_work_f = "/media/concha-eloko/Linux/PPT_clean" , path_ensemble_f = f"/media/concha-eloko/Linux/PPT_clean/ficheros_28032023/ensemble_1908") : 
	""" 
	Inputs : 
	Outputs : 

	Function that 
	
 
	"""
    # models object : 
    Dpo_classifier_models = {}
    hidden_channels = 1000
    conv = GATv2Conv
    heads = 1
    dropout = 0.1
    ensemble = {i : f"model_ratio_{i}" for i in ratios}
    for file in os.listdir(path_ensemble_f) : 
        if file[-2:] == "pt" and int(file.split(".")[3].split("Neg")[0]) in ensemble :
            ratio = int(file.split(".")[3].split("Neg")[0])
            model = Model(conv, hidden_channels, heads, dropout)
            model.load_state_dict(torch.load(f"{path_ensemble_f}/{file}"))
            Dpo_classifier_models[ensemble[ratio]] = model
    # Run the predictions :
    round_prediction = {}
    for ratio in ratios : 
        clean_results = {} 
        model = Dpo_classifier_models[f"model_ratio_{ratio}"]
        predictions, probabilities = make_predictions(model, graph)
        ids = get_nodes_id_single(graph[("B1", "infects", "A")].edge_label_index)
        results = tuple(zip(ids,predictions.numpy(),probabilities.numpy()))
        positive_results = [pred for pred in results if int(pred[1]) == 1]
        for pos_res in positive_results : 
            prot = pos_res[0][0]
            kltype = pos_res[0][1]
            score = pos_res[2]
            a = {}
            a[kltype] = score
            if score > 0.5 : 
                if prot not in clean_results : 
                    clean_results[prot] = a
                else :
                    clean_results[prot].update(a)
        for prot in clean_results :
            if prot not in round_prediction : 
                round_prediction[prot] = clean_results[prot]
            else :
                for kltype in clean_results[prot] :
                    if kltype not in round_prediction[prot] : 
                        round_prediction[prot][kltype] = clean_results[prot][kltype]
                    else :
                        round_prediction[prot][kltype] = round_prediction[prot][kltype] + clean_results[prot][kltype]
    return round_prediction

def format_predictions(round_prediction) :
	""" 
	Inputs : 
	Outputs : 

	Function that 
	
	"""
    final_results = {}
    for protein,hits in round_prediction.items() : 
        phage = protein.split("__")[0]
        if phage not in final_results : 
            tmp_hits = {}
            for kltype in hits : 
                if kltype in tmp_hits and hits[kltype] > tmp_hits[kltype]:
                    tmp_hits[kltype] = hits[kltype]
                elif kltype in tmp_hits and hits[kltype] < tmp_hits[kltype]:
                    pass
                elif kltype not in tmp_hits : 
                    tmp_hits[kltype] = hits[kltype]
            final_results[phage] = tmp_hits
        else :
            for kltype in hits : 
                if kltype in final_results[phage] and hits[kltype] > final_results[phage][kltype]:
                    final_results[phage][kltype] = hits[kltype]
                elif kltype in final_results[phage] and hits[kltype] < final_results[phage][kltype]:
                    pass
                elif kltype not in final_results[phage] : 
                    final_results[phage][kltype] = hits[kltype]
    return final_results


def get_top_n_kltypes(input_dict, n):
    output_dict = {}
    for key, sub_dict in input_dict.items():
        # Sort the sub_dict by values (in descending order) and get top n
        top_n_kl = sorted(sub_dict.items(), key=lambda x: x[1], reverse=True)[:n]
        # Add to output_dict
        output_dict[key] = top_n_kl
    return output_dict

def clean_print(dico) :
	pp = pprint.PrettyPrinter(width = 250, sort_dicts = True, compact = True)
	out = pp.pprint(dico)
	return out 

	
class CustomEncoder(json.JSONEncoder):
	""" 
	Inputs : 
	Outputs : 

	Function that 
	
 
	"""
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)