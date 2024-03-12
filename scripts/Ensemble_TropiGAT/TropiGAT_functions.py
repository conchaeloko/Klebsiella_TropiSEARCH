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
def make_query_graph(embeddings) :
    """
    This function builds the query graph for the ensemble model.
    Input : A list of the ESM2 embeddings of the depolymerase 
    Output : The query graph
    """
    query_graph = HeteroData()
    query_graph["B1"].x = torch.empty((1, 0))
    query_graph["B2"].x = torch.tensor(embeddings , dtype=torch.float)
    edge_index_B2_B1 = torch.tensor([[0,0]] , dtype=torch.long)
    query_graph['B2', 'expressed', 'B1'].edge_index = edge_index_B2_B1.t().contiguous()
    
    return query_graph
    
def make_ensemble_TropiGAT_old(path_ensemble) : 
	"""
	This function builds a dictionary with all the models that are part of the TropiGAT predictor
	Input : Path of the models
	Output : Dictionary
	# Make a json file with the versions of the GNN corresponding to each KL types
	# Load it
	# Create the correct model instance (TropiGAT_small_module or TropiGAT_big_module)
	"""
	DF_info = pd.read_csv(f"{path_work}/TropiGATv2.final_df.tsv", sep = "\t" ,  header = 0)
	DF_info_lvl_0 = DF_info[~DF_info["KL_type_LCA"].str.contains("\\|")]
	DF_info_lvl_0 = DF_info_lvl_0.drop_duplicates(subset = ["Infected_ancestor","index","prophage_id"] , keep = "first").reset_index(drop=True)
	df_prophages = DF_info_lvl_0.drop_duplicates(subset = ["Phage"])
	dico_prophage_count = dict(Counter(df_prophages["KL_type_LCA"]))
	dico_ensemble = {}
	for GNN_model in os.listdir(path_ensemble) :
		if GNN_model[-2:] == "pt" : 
			KL_type = GNN_model.split(".")[0]
			if dico_prophage_count[KL_type] >= 125 : 
				model = TropiGAT_models.TropiGAT_big_module(hidden_channels = 1280 , heads = 1)
			else :
				model = TropiGAT_models.TropiGAT_small_module(hidden_channels = 1280 , heads = 1)
			model.load_state_dict(torch.load(f"{path_ensemble}/{GNN_model}"))
			dico_ensemble[KL_type] = model
		
	return dico_ensemble

def make_ensemble_TropiGAT(path_ensemble) : 
	"""
	This function builds a dictionary with all the models that are part of the TropiGAT predictor
	Input : Path of the models
	Output : Dictionary
	# Make a json file with the versions of the GNN corresponding to each KL types
	# Load it
	# Create the correct model instance (TropiGAT_small_module or TropiGAT_big_module)
	"""
	errors = []
	DF_info = pd.read_csv(f"{path_work}/TropiGATv2.final_df_v2.filtered.tsv", sep = "\t" ,  header = 0)
	DF_info_lvl_0 = DF_info.copy()
	df_prophages = DF_info_lvl_0.drop_duplicates(subset = ["Phage"])
	dico_prophage_count = dict(Counter(df_prophages["KL_type_LCA"]))
	dico_ensemble = {}
	for GNN_model in os.listdir(path_ensemble) :
		if GNN_model[-2:] == "pt" :
			KL_type = GNN_model.split(".")[0]
			try :
				if dico_prophage_count[KL_type] >= 125 : 
					model = TropiGAT_models.TropiGAT_big_module(hidden_channels = 1280 , heads = 1)
				else :
					model = TropiGAT_models.TropiGAT_small_module(hidden_channels = 1280 , heads = 1)
				model.load_state_dict(torch.load(f"{path_ensemble}/{GNN_model}"))
				dico_ensemble[KL_type] = model
			except Exception as e :
				a = (KL_type , dico_prophage_count[KL_type], e)
				errors.append(a)
		
	return dico_ensemble , errors


def make_solo_TropiGAT(path_ensemble, target_KL_type) : 
	"""
	This function builds a dictionary with all the models that are part of the TropiGAT predictor
	Input : Path of the models
	Output : Dictionary
	# Make a json file with the versions of the GNN corresponding to each KL types
	# Load it
	# Create the correct model instance (TropiGAT_small_module or TropiGAT_big_module)
	"""
	errors = []
	DF_info = pd.read_csv(f"{path_work}/TropiGATv2.final_df_v2.filtered.tsv", sep = "\t" ,  header = 0)
	DF_info_lvl_0 = DF_info.copy()
	df_prophages = DF_info_lvl_0.drop_duplicates(subset = ["Phage"])
	dico_prophage_count = dict(Counter(df_prophages["KL_type_LCA"]))
	target_model = None
	for GNN_model in os.listdir(path_ensemble) :
		if GNN_model[-2:] == "pt" :
			KL_type = GNN_model.split(".")[0]
			if target_KL_type == KL_type :
				try :
					if dico_prophage_count[KL_type] >= 125 : 
						model = TropiGAT_models.TropiGAT_big_module(hidden_channels = 1280 , heads = 1)
					else :
						model = TropiGAT_models.TropiGAT_small_module(hidden_channels = 1280 , heads = 1)
					model.load_state_dict(torch.load(f"{path_ensemble}/{GNN_model}"))
					target_model = model
				except Exception as e :
					a = (KL_type , dico_prophage_count[KL_type], e)
					errors.append(a)
		
	return target_model , errors


def make_ensemble_TropiGAT_attention(path_ensemble) : 
	"""
	This function builds a dictionary with all the models that are part of the TropiGAT predictor
	Input : Path of the models
	Output : Dictionary , attention weights
	# Make a json file with the versions of the GNN corresponding to each KL types
	# Load it
	# Create the correct model instance (TropiGAT_small_module or TropiGAT_big_module)
	"""
	errors = []
	DF_info = pd.read_csv(f"{path_work}/TropiGATv2.final_df_v2.filtered.tsv", sep = "\t" ,  header = 0)
	DF_info_lvl_0 = DF_info.copy()
	df_prophages = DF_info_lvl_0.drop_duplicates(subset = ["Phage"])
	dico_prophage_count = dict(Counter(df_prophages["KL_type_LCA"]))
	dico_ensemble = {}
	for GNN_model in os.listdir(path_ensemble) :
		if GNN_model[-2:] == "pt" :
			KL_type = GNN_model.split(".")[0]
			try :
				if dico_prophage_count[KL_type] >= 125 : 
					model = TropiGAT_models.TropiGAT_big_module_attention(hidden_channels = 1280 , heads = 1)
				else :
					model = TropiGAT_models.TropiGAT_small_module_attention(hidden_channels = 1280 , heads = 1)
				model.load_state_dict(torch.load(f"{path_ensemble}/{GNN_model}"))
				dico_ensemble[KL_type] = model
			except Exception as e :
				a = (KL_type , dico_prophage_count[KL_type], e)
				errors.append(a)
		
	return dico_ensemble , errors


def make_ensemble_TropiSAGE(path_ensemble) : 
	"""
	This function builds a dictionary with all the models that are part of the TropiGAT predictor
	Input : Path of the models
	Output : Dictionary
	# Make a json file with the versions of the GNN corresponding to each KL types
	# Load it
	# Create the correct model instance (TropiGAT_small_module or TropiGAT_big_module)
	"""
	errors = []
	DF_info = pd.read_csv(f"{path_work}/TropiGATv2.final_df_v2.filtered.tsv", sep = "\t" ,  header = 0)
	DF_info_lvl_0 = DF_info.copy()
	df_prophages = DF_info_lvl_0.drop_duplicates(subset = ["Phage"])
	dico_prophage_count = dict(Counter(df_prophages["KL_type_LCA"]))
	dico_ensemble = {}
	for GNN_model in os.listdir(path_ensemble) :
		if GNN_model[-2:] == "pt" :
			KL_type = GNN_model.split(".")[0]
			try :
				if dico_prophage_count[KL_type] >= 125 : 
					model = TropiGAT_models.TropiGAT_big_sage_module(hidden_channels = 1280)
				else :
					model = TropiGAT_models.TropiGAT_small_sage_module(hidden_channels = 1280)
				model.load_state_dict(torch.load(f"{path_ensemble}/{GNN_model}"))
				dico_ensemble[KL_type] = model
			except Exception as e :
				a = (KL_type , dico_prophage_count[KL_type], e)
				errors.append(a)
		
	return dico_ensemble , errors

def make_ensemble_tailored_TropiGAT(path_ensemble , tailor_dico) : 
	"""
	This function builds a dictionary with all the models that are part of the TropiGAT predictor
	Input : Path of the models
	Output : Dictionary
	# Make a json file with the versions of the GNN corresponding to each KL types
	# Load it
	# Create the correct model instance (TropiGAT_small_module or TropiGAT_big_module)
	"""
	DF_info = pd.read_csv(f"{path_work}/TropiGATv2.final_df_v2.filtered.tsv", sep = "\t" ,  header = 0)
	#DF_info_lvl_0 = DF_info[~DF_info["KL_type_LCA"].str.contains("\\|")]
	#DF_info_lvl_0 = DF_info_lvl_0.drop_duplicates(subset = ["Infected_ancestor","index","prophage_id"] , keep = "first").reset_index(drop=True)
	DF_info_lvl_0 = DF_info.copy()
	df_prophages = DF_info_lvl_0.drop_duplicates(subset = ["Phage"])
	dico_prophage_count = dict(Counter(df_prophages["KL_type_LCA"]))
	dico_ensemble = {}
	for GNN_model in os.listdir(path_ensemble) :
		if GNN_model[-2:] == "pt" : 
			KL_type = GNN_model.split(".")[0]
			if KL_type not in tailor_dico : 
				if dico_prophage_count[KL_type] >= 125 : 
					model = TropiGAT_models.TropiGAT_big_module(hidden_channels = 1280 , heads = 1)
				else :
					model = TropiGAT_models.TropiGAT_small_module(hidden_channels = 1280 , heads = 1)
				model.load_state_dict(torch.load(f"{path_ensemble}/{GNN_model}"))
				dico_ensemble[KL_type] = model
			else : 
				if tailor_dico[KL_type] == "small" :
					model = TropiGAT_models.TropiGAT_small_module(hidden_channels = 1280 , heads = 1)
				else :
					model = TropiGAT_models.TropiGAT_big_module(hidden_channels = 1280 , heads = 1)
				model.load_state_dict(torch.load(f"{path_ensemble}/{GNN_model}"))
				dico_ensemble[KL_type] = model
	return dico_ensemble

@torch.no_grad()
def make_predictions(model, data):
	model.eval() 
	output = model(data)
	probabilities = torch.sigmoid(output)
	predictions = probabilities.round() 
	return predictions, round(probabilities.item() , 4) 
 
        
def run_prediction(query_graph, dico_ensemble) :
    dico_predictions = {}
    for KL_type in dico_ensemble :
        model = dico_ensemble[KL_type]
        prediction, probabilities = make_predictions(model, query_graph)
        if int(prediction) == 1 :
            dico_predictions[KL_type] = probabilities
        else :
            continue

    return dico_predictions

def format_predictions(predictions, sep = "__") : 
    final_results = {}
    for protein,hits in predictions.items() : 
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