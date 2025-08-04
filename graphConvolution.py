import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_scatter import scatter
import torch_sparse

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class UniGCNconv(Module):
    """
    a UniGCN convolution layer, similar to https://arxiv.org/abs/2105.00956
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(UniGCNconv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, args):
        V, E = args.V, args.E
        degV, degE = args.degV, args.degE
                
        X = torch.mm(input, self.weight)
        Xve = X[V]
        
        Xe = scatter(Xve, E, dim=0, reduce='mean').cuda()
                
        Xe = Xe * degE
        
        Xev = Xe[E]
        
        Xv = scatter(Xev, V, dim=0, reduce='sum', dim_size=args.num_node).cuda()
        
        X = Xv * degV
        
        if self.bias is not None:
            return X + self.bias
        else:
            return X
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               

class HGNNconv(Module):
    """
    a HGNN convolution layer, similar to https://arxiv.org/abs/1809.09401
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(HGNNconv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, args):
        V, E = args.V, args.E
                
        X = torch.mm(input, self.weight)
        Xve = X[V]
        
        Xe = scatter(Xve, E, dim=0, reduce='mean').cuda()
                
        Xev = Xe[E]
        
        Xv = scatter(Xev, V, dim=0, reduce='mean', dim_size=args.num_node).cuda()

        X = Xv 
        
        if self.bias is not None:
            return X + self.bias
        else:
            return X
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'