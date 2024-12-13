# %%
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import random
import numpy as np

import os
import pandas as pd
import sys
import torch

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

sys.path.append('/home/mrahma56/cs519/project')
from TAGLAS import get_dataset


from src.models import *




# DATASET = 'cora'
DATASET = 'wikics'
LLM_ID = "Llama-3"

dataset_key_dict = {
    'cora': 'cora_node',
    'wikics': 'wikics'
}


root_dir = "/home/mrahma56/cs519/project/"
taglas_dir = root_dir + "TAGLAS/"
llm_gen_dir = root_dir + "llm_gen_data/"
saved_model_dir = root_dir + "saved_models/"
embedding_dir = root_dir + "saved_embeddings/"

def load_taglas_dataset(dataset_key="cora_node", unlabel_ratio=None, print_info=True):
    # Load the dataset from TAGLAS
    dataset = get_dataset(dataset_key, root=taglas_dir)
    data = dataset._data

    # Set train, validation, and test masks based on the dataset key
    if dataset_key == "cora_node":
        data.train_lb_mask = dataset.side_data['node_split']['train'][0].clone()
        data.val_mask = dataset.side_data['node_split']['val'][0].clone()
        data.test_mask = dataset.side_data['node_split']['test'][0].clone()
    elif dataset_key == "wikics":
        data.train_lb_mask = dataset.side_data['node_split']['train'][:, 0].clone()
        data.val_mask = dataset.side_data['node_split']['val'][:, 0].clone()
        data.test_mask = dataset.side_data['node_split']['test'].clone()
    
    # Map labels and features
    data.y = data.label_map
    data.x_text = data.x
    data.x = data.x_original
    
    # Add num_classes to data
    data.num_classes = dataset.num_classes

    if unlabel_ratio:
        # Get indices of training nodes from the labeled training mask
        train_indices = data.train_lb_mask.nonzero(as_tuple=True)[0]
        
        # Get labels of training nodes
        train_labels = data.y[train_indices]
        
        # Initialize the mask for unlabeled training nodes
        data.train_ulb_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        
        class_label_counts = []  # Store labeled/unlabeled counts per class

        for cls in range(data.num_classes):
            # Get indices of training nodes belonging to the current class
            class_indices = train_indices[train_labels == cls]
            num_class_nodes = len(class_indices)
            
            # Calculate the number of nodes to unlabel (70%) and label (30%) for this class
            nodes_to_unlabel = int(unlabel_ratio * num_class_nodes)
            nodes_to_label = num_class_nodes - nodes_to_unlabel
            
            # Randomly select nodes to unlabel for this class
            unlabeled_indices = class_indices[torch.randperm(num_class_nodes)[:nodes_to_unlabel]]
            
            # Update the unlabeled mask
            data.train_ulb_mask[unlabeled_indices] = True
            
            # Count labeled and unlabeled samples for the class
            class_label_counts.append((cls, nodes_to_label, nodes_to_unlabel))
        
        # Update the labeled training mask
        data.train_lb_mask[data.train_ulb_mask] = False

    if print_info and unlabel_ratio:
        # Print the information about the unlabeled and labeled nodes
        print(f"Unlabeled ratio: {unlabel_ratio}")
        print(f"Labeled training nodes: {data.train_lb_mask.sum().item()}")
        print(f"Unlabeled training nodes: {data.train_ulb_mask.sum().item()}")
        
        # Print class-wise statistics
        print("\nClass-wise labeled and unlabeled counts:")
        for cls, num_labeled, num_unlabeled in class_label_counts:
            print(f"Class {cls}: Labeled = {num_labeled}, Unlabeled = {num_unlabeled}")
    
    # Retain only the required keys in the data object
    required_keys = [
        'x', 'y', 'train_lb_mask', 'train_ulb_mask', 'x_text',
        'val_mask', 'test_mask', 'num_classes', 
        'num_features', 'x_text', 'edge_index', 'edge_attr'
    ]
    for k in list(data.keys()):
        if k not in required_keys:
            data.pop(k)

    return data


# Initialize the NV-Embed-v2 model and tokenizer
model_name = "nvidia/NV-Embed-v2"  # Replace with the exact model name or path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

MAX_LEN = 5000
data = load_taglas_dataset(dataset_key_dict[DATASET], unlabel_ratio=0.7, print_info=False)

# Generate embeddings one by one with progress tracking
embeddings = []
for text in tqdm(data.x_text, desc="Generating Embeddings"):
    # Generate the embedding using model.encode()
    embedding = model.encode([text], instruction="", max_length=MAX_LEN).squeeze()
    embeddings.append(embedding)

# Convert list of embeddings to a tensor and store in data.x_embed
data.x_embed = torch.stack(embeddings)
torch.save(data.x_embed, os.path.join(embedding_dir,f'{dataset_key_dict[DATASET]}_{model_name.split('/')[-1]}.pt'))

# Example output
print(data.x_embed.shape)



