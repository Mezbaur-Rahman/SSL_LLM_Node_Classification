import os
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures
import sys
sys.path.append('/home/mrahma56/cs519/project')
from TAGLAS import get_dataset
from models import *

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataset utility function
def get_planetoid_dataset(name, normalize_features=True, split="public"):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', name)
    dataset = Planetoid(path, name, split=split, transform=NormalizeFeatures() if normalize_features else None)
    return dataset


# Function to relabel low-confidence samples
def llm_label_samples(low_conf_indices, data, dataset="cora", llm_id="Llama-3"):
    # print(f"Dataset: {dataset}")
    llm_gen_dir = "/home/mrahma56/cs519/project/llm_gen_data"
    num_classes = data.num_classes
    llm_gen_file = os.path.join(llm_gen_dir, f"{dataset}_{llm_id}.tsv")
    df = pd.read_csv(llm_gen_file, sep='\t')
    y_gen = torch.tensor(df['llm_label'].values)
    y_gen = torch.where((y_gen >= 0) & (y_gen < num_classes) , y_gen, torch.zeros_like(y_gen))
    return y_gen[low_conf_indices]

# GCN model
def create_gcn_model(num_features, hidden_dim, num_classes, dropout):
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, num_classes)
            self.dropout = dropout

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    return GCN()

# Training function
def train(model, data, optimizer):
    # print(data.device)
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_lb_mask], data.y[data.train_lb_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
def evaluate(model, data, mask_type):
    model.eval()
    mask = getattr(data, f'{mask_type}_mask')
    with torch.no_grad():
        logits = model(data)
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return loss, acc



def train_step_ssl(model, data, device, optimizer, dataset, llm_id="Llama-3", alpha=0.1, th=0.5, llm_label=False):
    """
    Performs a single training step for semi-supervised learning.

    Args:
        model: The graph neural network model.
        data: The graph data containing node features, edge indices, and masks.
        optimizer: Optimizer for updating model parameters.
        x: Node features.
        y: Labels for nodes.
        alpha: Weight for consistency loss.
        th: Confidence threshold for pseudo-labeling.
        llm_label: Boolean flag to trigger relabeling of low-confidence samples.

    Returns:
        num_low_conf_samples: Number of low-confidence samples.
        total_loss: Total training loss.
        labeled_loss: Loss on labeled samples.
        consistency_loss: Consistency loss for pseudo-labeled samples.
        updated_data: Updated graph data.
    """
    dataset_key = dataset_key_dict[dataset]
    # print(data.keys())
    # time.sleep(1000)
    
    model.train()
    optimizer.zero_grad()

    # Forward pass
    logits = model(data)
    out_prob = F.softmax(logits, dim=1)

    # Initialize losses
    labeled_loss = torch.tensor(0.0, device=device)
    consistency_loss = torch.tensor(0.0, device=device)
    num_low_conf_samples = 0

    # Compute labeled loss
    if data.train_lb_mask.sum() > 0:
        labeled_loss = F.nll_loss(logits[data.train_lb_mask], data.y[data.train_lb_mask])

    # Compute consistency loss and perform relabeling if enabled
    if data.train_ulb_mask.sum() > 0 and llm_label:
        pseudo_labels = out_prob[data.train_ulb_mask].argmax(dim=1)
        confidence_scores = out_prob[data.train_ulb_mask].max(dim=1).values
        confident_mask = confidence_scores > th
        low_conf_mask = ~confident_mask

        confident_indices = data.train_ulb_mask.nonzero(as_tuple=True)[0][confident_mask]
        low_conf_indices = data.train_ulb_mask.nonzero(as_tuple=True)[0][low_conf_mask]
        num_low_conf_samples = len(low_conf_indices)

        # Consistency loss for high-confidence samples
        if len(confident_indices) > 0:
            consistency_loss = F.nll_loss(logits[confident_indices], pseudo_labels[confident_mask])

        # Relabel low-confidence samples
        if len(low_conf_indices) > 0:
            new_labels = llm_label_samples(low_conf_indices, data, dataset=dataset_key, llm_id=llm_id)

            # Create new masks
            new_train_lb_mask = data.train_lb_mask.clone()
            new_train_ulb_mask = data.train_ulb_mask.clone()

            # Update masks and labels
            new_train_lb_mask[low_conf_indices] = True
            new_train_ulb_mask[low_conf_indices] = False
            data.y[low_conf_indices] = new_labels

            data.train_lb_mask = new_train_lb_mask
            data.train_ulb_mask = new_train_ulb_mask

    # Compute total loss and backpropagate
    total_loss = labeled_loss + alpha * consistency_loss
    total_loss.backward()
    optimizer.step()

    return (
        num_low_conf_samples,
        total_loss.item(),
        labeled_loss.item(),
        consistency_loss.item(),
        data
    )




def load_taglas_dataset(dataset_key="cora_node", unlabel_ratio=None,embedding_path=None, print_info=True):
    # Load the dataset from TAGLAS
    dataset = get_dataset(dataset_key, root="/home/mrahma56/cs519/project/TAGLAS/")
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
    # data.train_mask = data.train_lb_mask
    # Map labels and features
    data.y = data.label_map
    data.x_text = data.x
    data.x = data.x_original
    if embedding_path is not None and os.path.exists(embedding_path):
        print("Loading embedding from: ", embedding_path)
        data.x = torch.load(embedding_path)
    
    
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
        'x', 'y', 'train_lb_mask', 'train_ulb_mask', 
        'val_mask', 'test_mask', 'num_classes', 
        'num_features', 'x_text', 'edge_index', 'edge_attr'
    ]
    for k in list(data.keys()):
        if k not in required_keys:
            data.pop(k)
    # print(data.keys())
    return data


dataset_key_dict = {
    'cora': 'cora_node',
    'Cora': 'cora_node',
    'wikics': 'wikics'
}


def run_supervised(data, device, dataset="Cora",num_epochs=200,learning_rate=0.01,weight_decay=0.0005,hidden_channels=16,use_default_feats=True,use_taglas=False, print_logs=False):
    
    # dataset_key = dataset_key_dict[dataset]
    # if use_default_feats:
    #     embedding_path = None
    # else:
    #     embedding_path = f"/home/mrahma56/cs519/project/saved_embeddings/{dataset_key}_NV-Embed-v2.pt"
    
    # if use_taglas:
    #     data = load_taglas_dataset(dataset_key=dataset_key, unlabel_ratio=None,embedding_path=embedding_path, print_info=True)
    #     data = data.to(device)
    # else:
    #     dataset = get_planetoid_dataset(name=dataset, normalize_features=True, split="public")
    #     data = dataset[0]
    #     data.num_classes = dataset.num_classes
    #     data.num_node_features = dataset.num_node_features
    #     data= data.to(device)
    
    data= data.to(device)
    # model = create_gcn_model(num_features=data.num_node_features, hidden_dim=hidden_channels, num_classes=data.num_classes, dropout=0.5).to(device)
    model = GCN(num_features=data.num_features, hidden_dim=hidden_channels, num_classes=data.num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float('inf')
    best_test_acc = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, data, optimizer)
        val_loss, val_acc = evaluate(model, data, 'val')
        test_loss, test_acc = evaluate(model, data, 'test')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_acc = test_acc
        if print_logs:
            print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
    if print_logs:
        print(f"Best Validation Loss: {best_val_loss:.4f}, Best Test Accuracy: {best_test_acc:.4f}")

    return best_test_acc





def run_ssl(data, device, dataset="Cora",ulb_ratio=0.9,num_epochs=200,learning_rate=0.01,weight_decay=0.0005,hidden_channels=16,alpha=0.1, th=0.7, print_logs=False):
    
    # dataset_key = dataset_key_dict[dataset]
    # if use_default_feats:
    #     embedding_path = None
    # else:
    #     embedding_path = f"/home/mrahma56/cs519/project/saved_embeddings/{dataset_key}_NV-Embed-v2.pt"
    
    
    # data = load_taglas_dataset(dataset_key=dataset_key, unlabel_ratio=ulb_ratio, embedding_path=embedding_path, print_info=True)
    data = data.to(device)
    
    #print the device of data
    # print(data.device)
    # model = create_gcn_model(num_features=data.num_node_features, hidden_dim=hidden_channels, num_classes=data.num_classes, dropout=0.5).to(device)
    model = GCN(num_features=data.num_features, hidden_dim=hidden_channels, num_classes=data.num_classes).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float('inf')
    best_test_acc = 0

    for epoch in range(1, num_epochs + 1):
        low_conf_samples, total_loss, labeled_loss, const_loss, data = train_step_ssl(model, data, device, optimizer, dataset=dataset, alpha=alpha, th=th)
        val_loss, val_acc = evaluate(model, data, 'val')
        test_loss, test_acc = evaluate(model, data, 'test')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_acc = test_acc
        if print_logs:
            print(f"Epoch {epoch:03d}: Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    if print_logs:
        print(f"Best Validation Loss: {best_val_loss:.4f}, Best Test Accuracy: {best_test_acc:.4f}")

    return best_test_acc