"""
Created on 04/09/2023
TropiGAT functions
@author: conchaeloko
"""
# 0 - LIBRARIES
# --------------------------------------------------
import os
import json
import torch
import pprint
import random
import pandas as pd
import numpy as np
import logging
import warnings
from tqdm import tqdm
from collections import Counter
from itertools import product
from multiprocessing.pool import ThreadPool

# PyTorch Libraries
from torch_geometric.data import HeteroData, DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import to_hetero, HeteroConv, GATv2Conv
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import negative_sampling

# Sklearn Libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize, OneHotEncoder

# TropiGAT modules
import TropiGAT_graph
import TropiGAT_models


# 1 - GLOBAL VARIABLES
# --------------------------------------------------
path_work = "/media/concha-eloko/Linux/PPT_clean"
path_ensemble = f"{path_work}/ficheros_28032023/ensemble_1908"


# 2 - FUNCTIONS
# --------------------------------------------------
def load_dico_kltype(path):
	"""
	Input : path to the data
	Output : the dictionary with the one-hot-encoded KLtype to its value
	"""
	str_keys_dico_kltype = json.load(open(path))
	return {tuple(map(int, key.strip('()').split(','))): value for key, value in str_keys_dico_kltype.items()}

def get_nodes_id_single(graph_data, dpo_embeddings):
    """
	Inputs : 
	- graph_data : graph with the tested depolymerase
	- dpo_embeddings : dataframe of with the depolymerase name as index, ESM2 embeddings in iloc[1:1281]
	Output : A list of tuples with the (phage_id, kltype_id)
	"""
    dico_kltype = load_dico_kltype(f"{path_work}/tensorToKLtypes.json")
    B1A_index_file = tuple(zip(graph_data[("B1", "infects", "A")].edge_label_index[0].numpy(), graph_data[("B1", "infects", "A")].edge_label_index[1].numpy()))
    return [(dpo_embeddings.index[tup[0]], dico_kltype[tuple(graph_data["A"]["x"][tup[1]].numpy())]) for tup in B1A_index_file]

def graph_single_Dpo_pred(df_embeddings) : 
	"""
	Input : 
	- dpo_embeddings : dataframe of with the depolymerase name as index, ESM2 embeddings in iloc[1:1281]
	Output : A graph from the given dataframe embeddings
	"""
	# Load the template data :
	pred_data_single = torch.load(f'{path_work}/template_graph.KLtypes.pt')
	dico_kltype = load_dico_kltype(f"{path_work}/tensorToKLtypes.json")
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


@torch.no_grad()
def make_predictions(model, data):
	""" 
	Inputs : 
	- model : the GNN model used for the link predictions
	- data : the graph computed by graph_single_Dpo_pred
	Outputs : 
	the predictions and their corresponding probabilities  
	"""
	model.eval() 
	output = model(data)
	probabilities = torch.sigmoid(output)  # Convert output to probabilities
	predictions = probabilities.round()  # Convert probabilities to class labels
	return predictions, probabilities

		
def run_predictions(graph,dpo_embeddings ,ratios = [1,2,3,4,6,7]) : 
	""" 
	Inputs : 
	- graph : The graph to make predictions on 
	- dpo_embeddings : dataframe of with the depolymerase name as index, ESM2 embeddings in iloc[1:1281]
	- ratios : the different GNN to use for the predictions (default 1,2,3,4,6,7, i.e all with an MCC > 0.5)
	Outputs : A dictionary of the resulted probabilities for each positive predictions for a give item of the dpo_embeddings
	"""
	# ********************
	# The TropiGAT model : 
	class GNN(torch.nn.Module):
	    def __init__(self, edge_type , conv, hidden_channels, heads, dropout): 
	        super().__init__()
	        self.conv = conv((-1,-1), hidden_channels, add_self_loops = False, heads = heads, dropout = dropout, shared_weights = True)
	        self.hetero_conv = HeteroConv({edge_type: self.conv})
	    def forward(self, x_dict, edge_index_dict):
	        x = self.hetero_conv(x_dict, edge_index_dict)
	        return x
	class EdgeDecoder(torch.nn.Module):
	    def __init__(self, hidden_channels, heads):
	        super().__init__()
	        self.lin1 = torch.nn.Linear(heads*hidden_channels + 127, 512)
	        self.lin2 = torch.nn.Linear(512, 1)
	    def forward(self, x_dict_A , x_dict_B1, graph_data):
	        edge_type = ("B1", "infects", "A")
	        edge_feat_A = x_dict_A["A"][graph_data[edge_type].edge_label_index[1]]
	        edge_feat_B1 = x_dict_B1["B1"][graph_data[edge_type].edge_label_index[0]]
	        features_phage = torch.cat((edge_feat_A ,edge_feat_B1), dim=-1)
	        x = self.lin1(features_phage).relu()
	        x = self.lin2(x)
	        return x.view(-1)
	class Model(torch.nn.Module):
	    def __init__(self, conv, hidden_channels, heads, dropout):
	        super().__init__()
	        self.single_layer_model = GNN(("B2", "expressed", "B1") ,conv, hidden_channels,heads,dropout)
	        self.EdgeDecoder = EdgeDecoder(hidden_channels,heads)
	    def forward(self, graph_data):
	        b1_nodes = self.single_layer_model(graph_data.x_dict , graph_data.edge_index_dict)
	        a_nodes =  graph_data.x_dict
	        out = self.EdgeDecoder(a_nodes ,b1_nodes , graph_data)
	        return out
	# Model parameters : 
	Dpo_classifier_models = {}
	hidden_channels = 1000
	conv = GATv2Conv
	heads = 1
	dropout = 0.1
	ensemble = {i : f"model_ratio_{i}" for i in ratios}
	for file in os.listdir(path_ensemble) : 
		if file[-2:] == "pt" and int(file.split(".")[3].split("Neg")[0]) in ensemble :
			ratio = int(file.split(".")[3].split("Neg")[0])
			model = Model(conv, hidden_channels, heads, dropout)
			model.load_state_dict(torch.load(f"{path_ensemble}/{file}"))
			Dpo_classifier_models[ensemble[ratio]] = model
			
	# ********************
	# data :
	dico_kltype = load_dico_kltype(f"{path_work}/tensorToKLtypes.json")
	
	# ********************
	# Run the predictions :
	round_prediction = {}
	for ratio in ratios : 
		clean_results = {} 
		model = Dpo_classifier_models[f"model_ratio_{ratio}"]
		predictions, probabilities = make_predictions(model, graph)
		ids = get_nodes_id_single(graph ,dpo_embeddings)
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

def format_predictions(round_prediction , sep = "__") :
	""" 
	Inputs : 
	- The output of the function run_predictions
	- The separater of the phage name and the dpo name (default "__')
	Outputs : A dictionary of the results, with the data integrated for each phage
	"""
	final_results = {}
	try : 
		for protein,hits in round_prediction.items() : 
			phage = protein.split(sep)[0]
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
	except Exception as e:
		return "Mind the way the dpos were named."
		


def get_top_n_kltypes(input_dict, n):
	""" 
	Inputs : 
	- input_dict : the output of the format_predictions
	- n : the number of KLtype to return
	Outputs : a dictionary with the top n KLtypes and their scores
	"""
	output_dict = {}
	for key, sub_dict in input_dict.items():
		# Sort the sub_dict by values (in descending order) and get top n
		top_n_kl = sorted(sub_dict.items(), key=lambda x: x[1], reverse=True)[:n]
		# Add to output_dict
		output_dict[key] = top_n_kl
	return output_dict

def clean_print(dico) :
	""" 
	Inputs : a dico
	Outputs : pretty printed dico
	"""
	import pprint
	pp = pprint.PrettyPrinter(width = 150, sort_dicts = True, compact = True)
	out = pp.pprint(dico)
	return out 

	
class CustomEncoder(json.JSONEncoder):
	""" 
	For later ...
	"""
	def default(self, obj):
		if isinstance(obj, np.float32):
			return float(obj)
		return json.JSONEncoder.default(self, obj)