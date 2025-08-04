import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
import random
import copy
import os
import sys
import time
import argparse
import json
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import hyperparameters as hp
from torch.backends import cudnn
from utils import *
from graphConvolution import *
import torch.sparse 

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

SEED = 42
set_seed(SEED)

args = hp.parse_arguments()

def get_current_neighbors_sparse_gpu(cur_nodes, T_matrix_final):
    """Get current neighbors for sparse GPU computation"""
    if len(cur_nodes) == 0:
        return torch.zeros(T_matrix_final.size(1), device=T_matrix_final.device)
    
    if T_matrix_final.is_sparse:
        T_matrix_dense = T_matrix_final.to_dense()
    else:
        T_matrix_dense = T_matrix_final
        
    selected_rows = T_matrix_dense[cur_nodes]
    if selected_rows.dim() == 1:
        sum_result = selected_rows
    else:
        sum_result = selected_rows.sum(dim=0, keepdim=True)
    
    neighbors = (sum_result != 0).float()
    return neighbors

def compute_distance_matrix_gpu(features):
    """Compute distance matrix on GPU"""
    n = features.size(0)
    features_norm = torch.norm(features, dim=1, keepdim=True)
    dist = torch.mm(features, features.t())
    dist = features_norm.pow(2) + features_norm.t().pow(2) - 2 * dist
    dist = torch.clamp(dist, min=0.0)
    dist = torch.sqrt(dist)
    
    dist_min = dist.min()
    dist_max = dist.max()
    return (dist - dist_min) / (dist_max - dist_min)

def compute_balls_gpu(distance_matrix, radius):
    """Compute balls based on distance matrix and radius"""
    return (distance_matrix <= radius).float()

def compute_hedv_gpu(H, seed_nodes, beta=args.beta, device='cuda:0'):
    """Compute Hyperedge-based Diffusion Value (HEDV) on GPU"""
    if not torch.cuda.is_available() and device.startswith('cuda'):
        device = 'cpu'
        print("Warning: CUDA not available, switching to CPU computation")
        H = H.to(device)
    
    seed_nodes = torch.tensor(seed_nodes, dtype=torch.long, device=device)
    if len(seed_nodes) == 0:
        return 0.0
    seed_nodes = torch.unique(seed_nodes)
    K = seed_nodes.numel()
    if K == 0:
        return 0.0
    
    # Convert to float for calculations
    H_float = H.float()
    node_hyperedge_count = H_float.sum(dim=1)
    
    seed_H = H_float[seed_nodes]
    seed_hedges = torch.any(H[seed_nodes], dim=0)
    
    # Find neighbors and exclude seed nodes themselves
    neighbor_mask = torch.any(H[:, seed_hedges], dim=1)
    neighbor_mask[seed_nodes] = False  # Exclude seeds from neighbors
    neighbor_nodes = torch.where(neighbor_mask)[0]
    
    if neighbor_nodes.numel() == 0:
        return float(K)
    
    H_neighbors = H_float[neighbor_nodes]  # (num_neighbors, num_edges)
    
    # Vectorized computation of HEDV
    # 1. Compute common hyperedges between all neighbors and all seeds
    common_counts = torch.mm(H_neighbors, seed_H.t())  # (num_neighbors, K)
    
    # 2. Get hyperedge counts for seed nodes
    j_hedge_counts = node_hyperedge_count[seed_nodes]  # (K,)
    
    # 3. Compute edge probabilities
    # Add epsilon to avoid division by zero
    edge_probs = common_counts / (j_hedge_counts.unsqueeze(0) + 1e-8)  # (num_neighbors, K)
    
    # 4. Compute infection probabilities
    infection_probs = 1 - beta * edge_probs
    
    # 5. Compute joint infection probability. prod will ignore non-connected seeds (where prob is 1)
    joint_infections = torch.prod(infection_probs, dim=1)  # (num_neighbors,)
    
    # 6. Sum up HEDV
    hedv = K + torch.sum(1 - joint_infections)
    
    return hedv.cpu().item()

def compute_normalized_hedv(H, seed_nodes, max_hedv, beta=args.beta, device='cuda:0'):
    """Compute normalized HEDV"""
    hedv = compute_hedv_gpu(H, seed_nodes, beta, device)
    norm_hedv = hedv / (max_hedv + 1e-8)
    return norm_hedv

def compute_node_gain(node, cur_nodes, covered_balls, balls_dict, H_incident, num_node, max_hedv, beta=args.beta, device='cuda:0'):
    """Compute node gain combining ball coverage and HEDV gain"""
    # Ball coverage gain
    new_balls = covered_balls.union(balls_dict[node.item()])
    balls_gain = (len(new_balls) - len(covered_balls)) / num_node
    
    # Normalized HEDV gain
    current_seed_nodes = [node.item()] + cur_nodes
    norm_hedv_gain = (compute_hedv_gpu(H_incident, current_seed_nodes, beta, device) - \
                    compute_hedv_gpu(H_incident, cur_nodes, beta, device))/max_hedv
    
    # Combine gains using gamma weight
    gain = args.gamma * balls_gain + (1 - args.gamma) * norm_hedv_gain
    
    return gain

def select_nodes(features, H, device):
    """Select nodes using HIAL algorithm"""
    torch.cuda.empty_cache()
    H_incident = torch.BoolTensor(H.toarray()).to(device)
    num_node = H.shape[0]

    # Define validation and test indices
    idx_val = torch.arange(num_node - args.num_test - args.num_val, num_node - args.num_test, device=device)
    idx_test = torch.arange(num_node - args.num_test, num_node, device=device)
    idx_available = torch.tensor([i for i in range(num_node) if i not in idx_val and i not in idx_test], device=device)

    # Compute propagation matrix
    T = get_propagation_matrix(H, propagation_name=args.use_propagation, alpha=args.alpha)
    T_sparse = T.tocoo()
    indices = torch.LongTensor(np.array([T_sparse.row, T_sparse.col]))
    values = torch.FloatTensor(T_sparse.data)
    T_matrix = torch.sparse_coo_tensor(indices, values, T_sparse.shape).to(device)
    
    # Multi-layer propagation
    T_matrix_final = T_matrix.clone()
    for i in range(args.prop_layer - 1):
        try:
            T_matrix_final = torch.sparse.mm(T_matrix_final, T_matrix)
        except:
            T_matrix_final = torch.mm(T_matrix_final.to_dense(), T_matrix.to_dense())
    
    # Propagate features
    features = features.to(device)
    try:
        features_final = torch.sparse.mm(T_matrix_final, features)
    except:
        features_final = torch.mm(T_matrix_final.to_dense(), features)

    # Compute distance matrix and balls
    with torch.no_grad():
        distance_final = compute_distance_matrix_gpu(features_final)
        balls = compute_balls_gpu(distance_final, args.radium)

    # Build balls dictionary
    balls_dict = {}
    for node in idx_available:
        neighbors_tmp = get_current_neighbors_sparse_gpu([node], T_matrix_final)
        dot_result = torch.matmul(balls, neighbors_tmp)
        tmp_set = set(torch.where(dot_result != 0)[0].cpu().numpy())
        balls_dict[node.item()] = tmp_set

    # Pre-compute max HEDV for normalization
    max_hedv = compute_hedv_gpu(H_incident, list(range(num_node)), beta=args.beta, device=device)

    # Greedy node selection
    count = 0
    idx_train = []
    covered_balls = set()
    
    idx_available_tmp = idx_available.clone()
    
    while count < args.num_coreset:
        max_gain = -float('inf')
        node_max = None
        
        for node in idx_available_tmp:
            gain = compute_node_gain(
                node, idx_train, covered_balls, balls_dict,
                H_incident, num_node, max_hedv,
                beta=args.beta, device=device
            )
            
            if gain > max_gain:
                max_gain = gain
                node_max = node
        
        idx_train.append(node_max.item())
        idx_available_tmp = idx_available_tmp[idx_available_tmp != node_max]
        covered_balls = covered_balls.union(balls_dict[node_max.item()])
        count += 1
        
        if count % 20 == 0:
            print(f"Selected {count} nodes")

    return idx_train

class HGNN(nn.Module):
    """Hypergraph Neural Network model"""
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(HGNN, self).__init__()
        
        self.gc1 = HGNNconv(nfeat, nhid, bias=True)
        self.gc2 = HGNNconv(nhid, nclass, bias=True)
        self.dropout = dropout
        
    def forward(self, x, args):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, args))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, args)
        return x

class EarlyStopping:
    """Early stopping mechanism to prevent overfitting"""
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        
    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def train(epoch, model, record):
    """Training function for one epoch"""
    model.train()
    optimizer.zero_grad()
    output = model(features_GCN, args)

    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(features_GCN, args)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    early_stopping(loss_val)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    
    if early_stopping.early_stop or epoch == 399:
        global is_last_epoch
        is_last_epoch = 1
        record[acc_val.item()] = acc_test.item()
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("---------------------------------------------------")
    print(f"--Device: {device}")
    print(f'--Dataset: {args.use_dataset}')
    print(f'--Propagation: {args.use_propagation}')
    print(f'--prop_layer: {str(args.prop_layer)}')
    print("---------------------------------------------------")
    # Load data with split support
    features, labels, H = load_data(dataset=args.use_dataset)
    num_labels = labels.max().item() + 1
    args.num_coreset = num_labels * args.K

    # Setup hypergraph parameters
    (V, E), value = torch_sparse.from_scipy(H)
    args.V, args.E = V.cuda(), E.cuda()
    degV = torch.from_numpy(H.sum(1)).view(-1, 1).float()
    degE = scatter(degV[V], E, dim=0, reduce='mean')
    args.degE = degE.pow(-0.5).cuda()
    degV = degV.pow(-0.5)
    degV[degV.isinf()] = 1
    args.degV = degV.cuda()
    num_node = features.shape[0]
    args.num_node = num_node

    # Select training nodes using HIAL
    print('Start selecting nodes:')
    time_select_start = time.time()
    idx_train = select_nodes(features, H, device)
    time_select_end = time.time()
    print(f'Select nodes time(s): {time_select_end - time_select_start}')
    print("---------------------------------------------------")
    idx_train = torch.LongTensor(idx_train).cuda()

    # Load split data to get proper indices
    split_data_path = f"./data/{args.use_dataset}/split_data.pickle"
    if os.path.exists(split_data_path):
        with open(split_data_path, 'rb') as f:
            split_data = pickle.load(f)
        
        # Use precomputed indices
        train_indices = split_data['train_indices']
        val_indices = split_data['val_indices']
        test_indices = split_data['test_indices']
        
        # Convert to tensors and move to GPU
        idx_train = torch.LongTensor(train_indices).cuda()
        idx_val = torch.LongTensor(val_indices).cuda()
        idx_test = torch.LongTensor(test_indices).cuda()
        
    else:
        # Fallback to original logic
        idx_val = torch.LongTensor(range(num_node - args.num_test - args.num_val, num_node - args.num_test)).cuda()
        idx_test = torch.LongTensor(range(num_node - args.num_test, num_node)).cuda()


    # Prepare features and labels
    features_GCN = copy.deepcopy(features)
    features_GCN = torch.FloatTensor(features_GCN).cuda()
    labels = labels.cuda()

    print('Evaluation begin')
    time_train_start = time.time()
    record = {}
    is_last_epoch = 0
    print(f'--Repeat training turns: {args.train_turns}')
    print(f'--Size of seed set: {args.num_coreset}')
    print(f'--Size of validation set: {len(idx_val)}')
    print(f'--Size of test set: {len(idx_test)}')
    print(f'--Hidden size: {args.hidden_size}')
    print(f'--Dropout rate: {args.dropout}')
    print(f'--Learning rate: {args.lr}')
    print(f'--Weight decay: {args.weight_decay}')
    print("---------------------------------------------------")

    # Training loop
    for i in range(args.train_turns):
        if (i+1) % 20 == 0:
            print(f'HIAL training {i+1} turns end.')
        is_last_epoch = 0
        model = HGNN(nfeat=features_GCN.shape[1],
                nhid=args.hidden_size,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
        model.cuda()
        early_stopping = EarlyStopping(patience = 10)
        optimizer = optim.Adam(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(400):
            train(epoch,model,record)
            if is_last_epoch:
                break
    time_train_end = time.time()
    print(f'train time(s): {time_train_end - time_train_start}')
    print("---------------------------------------------------")
    # Calculate final accuracy
    bit_list = sorted(record.keys())
    bit_list.reverse()
    total_accuracy = list()
    for key in bit_list[:10]:
        value = record[key]
        total_accuracy.append(value)
    print(f'average_test_accuracy ± std :{np.mean(total_accuracy)} ± {np.std(total_accuracy)}')
