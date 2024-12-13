# Necessary imports
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import nn



# Graph Convolutional Cluster (GCC) model
class GCC(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCC, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x




class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# class GCN(torch.nn.Module):
#     def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5, num_layers=3, norm=None) -> None:
#         super().__init__()
#         self.convs = torch.nn.ModuleList()
#         self.norms = torch.nn.ModuleList()
#         self.num_layers = num_layers
#         self.dropout = dropout
#         if num_layers == 1:
#             self.convs.append(GCNConv(num_features, num_classes, cached=False, normalize=True))
#         else:
#             self.convs.append(GCNConv(num_features, hidden_channels, cached=False, normalize=True))
#             if norm:
#                 self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
#             else:
#                 self.norms.append(torch.nn.Identity())

#             for _ in range(num_layers - 2):
#                 self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False, normalize=True))
#                 if norm:
#                     self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
#                 else:
#                     self.norms.append(torch.nn.Identity())

#             self.convs.append(GCNConv(hidden_channels, num_classes, cached=False, normalize=True))

#     def forward(self, x, edge_index, edge_weight=None):
#         # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
#         for i in range(self.num_layers):
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             if edge_weight is not None:
#                 x = self.convs[i](x, edge_index, edge_weight)
#             else:
#                 x = self.convs[i](x, edge_index)
#             if i != self.num_layers - 1:
#                 x = self.norms[i](x)
#                 x = F.relu(x)
#         return x


# class GCN(torch.nn.Module):
#     def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5, num_layers=3, norm=None):
#         super().__init__()
#         self.convs = torch.nn.ModuleList()
#         self.norms = torch.nn.ModuleList()
#         self.num_layers = num_layers
#         self.dropout = dropout

#         if num_layers == 1:
#             self.convs.append(GCNConv(num_features, num_classes, cached=False, normalize=True))
#         else:
#             self.convs.append(GCNConv(num_features, hidden_channels, cached=False, normalize=True))
#             if norm:
#                 self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
#             else:
#                 self.norms.append(torch.nn.Identity())

#             for _ in range(num_layers - 2):
#                 self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False, normalize=True))
#                 if norm:
#                     self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
#                 else:
#                     self.norms.append(torch.nn.Identity())

#             self.convs.append(GCNConv(hidden_channels, num_classes, cached=False, normalize=True))

#     def forward(self, x, edge_index):
#         for i in range(self.num_layers):
#             x = self.convs[i](x, edge_index)  # Apply convolution
#             if i != self.num_layers - 1:      # Apply normalization and activation except for the last layer
#                 x = self.norms[i](x)
#                 x = F.relu(x)
#                 x = F.dropout(x, p=self.dropout, training=self.training)
#         return x
