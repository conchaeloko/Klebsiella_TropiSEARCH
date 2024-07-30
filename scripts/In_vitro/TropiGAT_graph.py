"""
Created on 25/09/2023
TropiGAT graph functions
@author: conchaeloko
"""
# 0 - LIBRARIES
# --------------------------------------------------
import torch
import random
import pandas as pd
import numpy as np
import logging
import warnings
from tqdm import tqdm

# PyTorch Libraries
from torch_geometric.data import HeteroData

# Sklearn Libraries
from sklearn.preprocessing import LabelEncoder , label_binarize , OneHotEncoder

def build_graph_baseline(df_info) : 
    # **************************************************************
    # initialize the graph
    graph_data = HeteroData()
    # Indexation process  
    indexation_nodes_A = df_info["Infected_ancestor"].unique().tolist()  
    indexation_nodes_B1 = df_info["Phage"].unique().tolist()
    indexation_nodes_B2 = df_info["index"].unique().tolist() 
    ID_nodes_A = {item:index for index, item in enumerate(indexation_nodes_A)}
    ID_nodes_A_r = {index:item for index, item in enumerate(indexation_nodes_A)}
    ID_nodes_B1 = {item:index for index, item in enumerate(indexation_nodes_B1)}
    ID_nodes_B1_r = {index:item for index, item in enumerate(indexation_nodes_B1)}
    ID_nodes_B2 = {item:index for index, item in enumerate(indexation_nodes_B2)}
    ID_nodes_B2_r = {index:item for index, item in enumerate(indexation_nodes_B2)}
    # **************************************************************
    # Make the node feature file : 
    OHE = OneHotEncoder(sparse=False)
    one_hot_encoded = OHE.fit_transform(df_info[["KL_type_LCA"]])
    label_mapping = {label: one_hot_encoded[i] for i, label in enumerate(OHE.categories_[0])}
    embeddings_columns = [str(i) for i in range(1, 1281)]
    node_feature_A = torch.tensor([label_mapping[df_info[df_info["Infected_ancestor"] == ID_nodes_A_r[i]]["KL_type_LCA"].values[0]] for i in range(0,len(ID_nodes_A_r))], dtype=torch.float)
    node_feature_B1 = torch.zeros((len(ID_nodes_B1), 0), dtype=torch.float)
    node_feature_B2 = torch.tensor([df_info[df_info["index"] == ID_nodes_B2_r[i]][embeddings_columns].values[0].tolist() for i in range(0,len(ID_nodes_B2_r))] , dtype=torch.float)
    # feed the graph
    graph_data["A"].x = node_feature_A
    graph_data["B1"].x = node_feature_B1
    graph_data["B2"].x = node_feature_B2
    # **************************************************************
    # Make edge file
    # Node B1 (prophage) - Node A (bacteria) :
    edge_index_B1_A = []
    track_B1_A = set()
    for _, row in df_info.iterrows() :
        pair = [ID_nodes_B1[row["Phage"]], ID_nodes_A[row["Infected_ancestor"]]]
        if tuple(pair) not in track_B1_A : 
            track_B1_A.add(tuple(pair))
            edge_index_B1_A.append(pair)
        else :
            continue
    edge_index_B1_A = torch.tensor(edge_index_B1_A , dtype=torch.long)
    # Node A (bacteria) - Node B1 (prophage) :
    edge_index_A_B1 = []
    track_A_B1 = set()
    for _, row in df_info.iterrows() :
        pair = [ID_nodes_A[row["Infected_ancestor"]] , ID_nodes_B1[row["Phage"]]]
        if tuple(pair) not in track_A_B1 :
            track_A_B1.add(tuple(pair))
            edge_index_A_B1.append(pair)
    edge_index_A_B1 = torch.tensor(edge_index_A_B1 , dtype=torch.long)
    # Node B2 (depolymerase) - Node B1 (prophage) :
    edge_index_B2_B1 = []
    for phage in df_info.Phage.unique() :
        all_data_phage = df_info[df_info["Phage"] == phage]
        for _, row in all_data_phage.iterrows() :
            edge_index_B2_B1.append([ID_nodes_B2[row["index"]], ID_nodes_B1[row["Phage"]]])
    edge_index_B2_B1 = torch.tensor(edge_index_B2_B1 , dtype=torch.long)
    # feed the graph
    graph_data['B1', 'infects', 'A'].edge_index = edge_index_B1_A.t().contiguous()
    graph_data['B2', 'expressed', 'B1'].edge_index = edge_index_B2_B1.t().contiguous()
    # That one is optional  
    graph_data['A', 'harbors', 'B1'].edge_index = edge_index_A_B1.t().contiguous()
    dico_prophage_kltype_associated = {}
    for negative_index,phage in tqdm(enumerate(df_info["Phage"].unique().tolist())) :
        kltypes = set()
        dpos = df_info[df_info["Phage"] == phage]["index"]
        for dpo in dpos : 
            tmp_kltypes = df_info[df_info["index"] == dpo]["KL_type_LCA"].values
            kltypes.update(tmp_kltypes)
        dico_prophage_kltype_associated[phage] = kltypes
    return graph_data , dico_prophage_kltype_associated


def build_graph_masking(graph_data_input, dico_prophage_kltype_associated , df_info, KL_type, ratio , f_train, f_test, f_eval) : 
    # **************************************************************
    # Indexation process  
	graph_data = graph_data_input.clone()
	indexation_nodes_A = df_info["Infected_ancestor"].unique().tolist()  
	indexation_nodes_B1 = df_info["Phage"].unique().tolist()
	indexation_nodes_B2 = df_info["index"].unique().tolist() 
	ID_nodes_A = {item:index for index, item in enumerate(indexation_nodes_A)}
	ID_nodes_A_r = {index:item for index, item in enumerate(indexation_nodes_A)}
	ID_nodes_B1 = {item:index for index, item in enumerate(indexation_nodes_B1)}
	ID_nodes_B1_r = {index:item for index, item in enumerate(indexation_nodes_B1)}
	ID_nodes_B2 = {item:index for index, item in enumerate(indexation_nodes_B2)}
	ID_nodes_B2_r = {index:item for index, item in enumerate(indexation_nodes_B2)}
	# **************************************************************
	# Make the Y file : 
	B1_labels = df_info.drop_duplicates(subset = ["Phage"], keep = "first")["KL_type_LCA"].apply(lambda x : 1 if x == KL_type else 0).to_list()
	graph_data["B1"].y = torch.tensor(B1_labels)
	# **************************************************************
	# Make mask files :
	# get the positive and negative indices lists :
	positive_indices = [index for index,label in enumerate(B1_labels) if label==1]
	negative_indices = []
	for negative_index,phage in enumerate(df_info["Phage"].unique().tolist()) :
		if KL_type not in dico_prophage_kltype_associated[ID_nodes_B1_r[negative_index]] :
			negative_indices.append(negative_index)
	# make the train, test, val lists : 
	n_samples = len(positive_indices)
	#train_indices, test_indices, val_indices = [],[],[]
	# make train : 
	train_pos = random.sample(positive_indices, int(f_train*n_samples))
	train_neg = random.sample(negative_indices, int(f_train*n_samples*ratio))
	train_indices = train_pos + train_neg
	train_mask = [True if n in train_indices else False for n in range(0,len(B1_labels))]
	# make test : 
	pool_positives_test = list(set(positive_indices) - set(train_pos))
	pool_negatives_test = list(set(negative_indices) - set(train_neg))
	test_pos = random.sample(pool_positives_test, int(f_test*n_samples))
	test_neg = random.sample(pool_negatives_test, int(f_test*n_samples*ratio))
	test_indices = test_pos + test_neg
	test_mask = [True if n in test_indices else False for n in range(0,len(B1_labels))]
	# make eval
	pool_positives_eval = list(set(positive_indices) - set(train_pos) - set(test_pos))
	pool_negatives_eval = list(set(negative_indices) - set(train_neg) - set(test_neg))
	eval_pos = random.sample(pool_positives_eval, int(f_eval*n_samples))
	eval_neg = random.sample(pool_negatives_eval, int(f_eval*n_samples*ratio))
	eval_indices = eval_pos + eval_neg
	eval_mask = [True if n in eval_indices else False for n in range(0,len(B1_labels))]
	# Transfer data to graph :
	graph_data["B1"].train_mask = torch.tensor(train_mask)
	graph_data["B1"].test_mask = torch.tensor(test_mask)
	graph_data["B1"].eval_mask = torch.tensor(eval_mask)
	
	return graph_data


def build_graph_masking_v2(graph_data_input, dico_prophage_kltype_associated , df_info, KL_type, ratio , f_train, f_test, f_eval, seed = 1) : 
    # **************************************************************
    # Indexation process  
	graph_data = graph_data_input.clone()
	indexation_nodes_A = df_info["Infected_ancestor"].unique().tolist()  
	indexation_nodes_B1 = df_info["Phage"].unique().tolist()
	indexation_nodes_B2 = df_info["index"].unique().tolist() 
	ID_nodes_A = {item:index for index, item in enumerate(indexation_nodes_A)}
	ID_nodes_A_r = {index:item for index, item in enumerate(indexation_nodes_A)}
	ID_nodes_B1 = {item:index for index, item in enumerate(indexation_nodes_B1)}
	ID_nodes_B1_r = {index:item for index, item in enumerate(indexation_nodes_B1)}
	ID_nodes_B2 = {item:index for index, item in enumerate(indexation_nodes_B2)}
	ID_nodes_B2_r = {index:item for index, item in enumerate(indexation_nodes_B2)}
	# **************************************************************
	# Make the Y file : 
	B1_labels = df_info.drop_duplicates(subset = ["Phage"], keep = "first")["KL_type_LCA"].apply(lambda x : 1 if x == KL_type else 0).to_list()
	graph_data["B1"].y = torch.tensor(B1_labels)
	# **************************************************************
	# Make mask files :
	# define the seed :
	random.seed(seed)
	# get the positive and negative indices lists :
	positive_indices = [index for index,label in enumerate(B1_labels) if label==1]
	negative_indices = []
	for negative_index,phage in enumerate(df_info["Phage"].unique().tolist()) :
		if KL_type not in dico_prophage_kltype_associated[ID_nodes_B1_r[negative_index]] :
			negative_indices.append(negative_index)
	# make the train, test, val lists : 
	n_samples = len(positive_indices)
	#train_indices, test_indices, val_indices = [],[],[]
	# make train : 
	train_pos = random.sample(positive_indices, int(f_train*n_samples))
	train_neg = random.sample(negative_indices, int(f_train*n_samples*ratio))
	train_indices = train_pos + train_neg
	train_mask = [True if n in train_indices else False for n in range(0,len(B1_labels))]
	# make test : 
	pool_positives_test = list(set(positive_indices) - set(train_pos))
	pool_negatives_test = list(set(negative_indices) - set(train_neg))
	test_pos = random.sample(pool_positives_test, int(f_test*n_samples))
	test_neg = random.sample(pool_negatives_test, int(f_test*n_samples*ratio))
	test_indices = test_pos + test_neg
	test_mask = [True if n in test_indices else False for n in range(0,len(B1_labels))]
	# make eval
	pool_positives_eval = list(set(positive_indices) - set(train_pos) - set(test_pos))
	pool_negatives_eval = list(set(negative_indices) - set(train_neg) - set(test_neg))
	eval_pos = random.sample(pool_positives_eval, int(f_eval*n_samples))
	eval_neg = random.sample(pool_negatives_eval, int(f_eval*n_samples*ratio))
	eval_indices = eval_pos + eval_neg
	eval_mask = [True if n in eval_indices else False for n in range(0,len(B1_labels))]
	# Transfer data to graph :
	graph_data["B1"].train_mask = torch.tensor(train_mask)
	graph_data["B1"].test_mask = torch.tensor(test_mask)
	graph_data["B1"].eval_mask = torch.tensor(eval_mask)
	
	return graph_data
