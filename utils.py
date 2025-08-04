import sys
import numpy as np
import scipy.sparse as sp
import pickle
import torch
import torch.nn.functional
import torch_sparse
from torch_scatter import scatter


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # prevent division by zero
    rowsum[rowsum == 0] = 1
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def get_edge_embedding(H, features):
    """Calculate the edge embedding"""
    N = sparse_mx_to_torch_sparse_tensor(H.dot(H.T)).to_dense().cuda()  # V x V
    H_ = sparse_mx_to_torch_sparse_tensor(H.T).to_dense().cuda()  # V x E
    mask = ~torch.eye(N.shape[0], dtype=torch.bool) + 0
    mask = mask.cuda()
    N_sum = torch.sum(N * mask, dim=1) 
    neighbor = H_ * N_sum
    row_sum = torch.sum(neighbor, dim=1, keepdim=True).cuda()
    row_sum[row_sum == 0] = 1
    features = features.cuda()
    return (neighbor / row_sum).dot(features)   
    
def get_propagation_matrix(H, propagation_name, alpha=0.5):
    """Calculate the propagation matrix"""
    # degree of nodes
    deg_V = torch.from_numpy(H.sum(1)).view(-1, 1).float()
    # degree of hyperedge
    deg_E = torch.from_numpy(H.sum(0)).view(-1, 1).float()

    (row, col), value = torch_sparse.from_scipy(H)
    V, E = row, col
    
    # average degree of hyperedge
    deg_AVE = scatter(deg_V[V], E, dim=0, reduce="mean") 
    
    deg_E1 = deg_E.pow(-1)
    deg_E2 = deg_E.pow(-0.5)
    deg_V1 = deg_V.pow(-1)
    deg_V2 = deg_V.pow(-0.5)
    
    deg_AVE = deg_AVE.pow(-0.5)
    
    deg_V1[deg_V1.isinf()] = 1 # when not added self-loop, some nodes might not be connected with any edge
    deg_V2[deg_V2.isinf()] = 1

    diag_E1 = sp.diags(deg_E1.flatten().cpu().numpy())
    diag_E = sp.diags(deg_E.flatten().cpu().numpy())
    diag_E2 = sp.diags(deg_E2.flatten().cpu().numpy())
    diag_V1 = sp.diags(deg_V1.flatten().cpu().numpy())
    diag_V2 = sp.diags(deg_V2.flatten().cpu().numpy())
    diag_AVE = sp.diags(deg_AVE.flatten().cpu().numpy())
    if propagation_name == "HGNN":
        return diag_V2.dot(H).dot(diag_E1).dot(H.T).dot(diag_V2)
    
    elif propagation_name == "UniGCN":
        return diag_V2.dot(H).dot(diag_AVE).dot(diag_E1).dot(H.T)
    
    elif propagation_name == "VilLain":
        return diag_V1.dot(H).dot(diag_E1).dot(H.T)
    
    elif propagation_name == "AHGAE":
        return 0.1 * sp.eye(H.shape[0]) + 0.9 * diag_V2.dot(H).dot(diag_E1).dot(H.T).dot(diag_V2)
    
    elif propagation_name == "HOIK":     
        # 1. Use LIL format for initial construction
        K = H.dot(diag_E - sp.eye(H.shape[1])).dot(H.T).tolil()
        
        # 2. Directly use diagonal operation in LIL format
        K.setdiag(0)
        
        # 3. Calculate degree matrix
        D_K = torch.tensor(K.sum(axis=1).A1).float()
        D_K = D_K.pow(-1/2)
        D_K = sp.diags(D_K.numpy())
        
        # 4. Choose the most suitable format for the operation
        # For matrix multiplication, use CSR format
        K = K.tocsr()
        D_K = D_K.tocsr()
        
        # 5. Use dot method for matrix multiplication
        result = alpha * D_K.dot(D_K.dot(K)) + (1 - alpha) * sp.eye(H.shape[0])
        return result


def incident_to_adjacency(H):
    """
    Returns:
        scipy.sparse.csr_matrix: Adjacency matrix, shape (N, N)
    """
    if not sp.issparse(H):
        raise TypeError("Input must be a sparse matrix")
    
    # Ensure input is in CSR format
    H = H.tocsr()
    N, M = H.shape
    
    # Use a more efficient way to build the adjacency matrix
    # For each hyperedge, compute all pairs of nodes connected by it
    rows, cols = [], []
    
    # Iterate over each hyperedge
    for e in range(M):
        # Get all nodes in the current hyperedge
        nodes = H[:, e].nonzero()[0]
        n = len(nodes)
        
        # Create connections for each pair of nodes in the hyperedge
        for i in range(n):
            for j in range(i + 1, n):
                rows.extend([nodes[i], nodes[j]])
                cols.extend([nodes[j], nodes[i]])
    
    # Create adjacency matrix
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    
    # Convert to CSR format and return
    return A.tocsr()
    
def load_data(path="./data", dataset="Cora", add_self_loop=True):
    names = ['features', 'labels', 'hypergraph']
    objects = []
    
    for i in range(len(names)):
        with open(f"{path}/{dataset}/{names[i]}.pickle", 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pickle.load(f, encoding='latin1'))
            else:
                objects.append(pickle.load(f))

    features, labels, hypergraph = tuple(objects)
       
    # process features:
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    
    # process labels:
    labels = torch.LongTensor(labels)
    
    # process hypergraph:
    # add self-loop
    if add_self_loop:
        Vs = set(range(features.shape[0]))

        # only add self-loop to those that are originally un-self-looped
        # TODO: maybe we should remove some repeated self-loops?
        for edge, nodes in hypergraph.items():
            if len(nodes) == 1 and nodes[0] in Vs:
                Vs.remove(nodes[0])

        for v in Vs:
            hypergraph[f'self-loop-{v}'] = [v]
    # convert hypergraph to sparse matrix
    N, M = features.shape[0], len(hypergraph)
    indptr, indices, data = [0], [], []
    for e, vs in hypergraph.items():
        indices += vs
        data += [1] * len(vs)
        indptr.append(len(indices))
    H = sp.csc_matrix((data, indices, indptr), shape=(N, M), dtype=int).tocsr() # V x E
    return features, labels, H

def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()