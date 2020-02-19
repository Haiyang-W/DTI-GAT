import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from gat_layers import GraphAttentionLayer, SpGraphAttentionLayer
from torch_geometric.nn import GATConv, GCNConv

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, noutput, dropout, negative_slope, nheads):
        """ version of GAT."""
        super(GAT, self).__init__()
        # self.dropout = 0.6
        self.dropout = dropout
        self.attentions = GATConv(nfeat, nhid, nheads, True, negative_slope=negative_slope, dropout=self.dropout)
        self.out_att = GATConv(nhid*nheads, noutput, 1, False, negative_slope=negative_slope, dropout=self.dropout)
        self.BatchNorm = torch.nn.BatchNorm1d(num_features=noutput)
        self.LayerNorm = torch.nn.LayerNorm(noutput)

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.attentions(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, edge_index))
        x = self.BatchNorm(x)
        x = self.LayerNorm(x)
        return x

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, noutput, dropout):
        """version of GCN."""
        super(GCN, self).__init__()
        # self.dropout = 0.6
        self.dropout = dropout
        self.gcn1 = GCNConv(nfeat, nhid)

        self.gcn2 = GCNConv(nhid, noutput)
        self.BatchNorm = torch.nn.BatchNorm1d(num_features=noutput)
        self.LayerNorm = torch.nn.LayerNorm(noutput)

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.gcn1(x, edge_index, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.gcn2(x, edge_index, edge_weight))
        x = self.BatchNorm(x)
        x = self.LayerNorm(x)
        return x

class NN(nn.Module):
    def __init__(self, ninput, nhidden, noutput, nlayers, dropout=0.3):
        """
        """
        super(NN, self).__init__()
        self.dropout = dropout
        self.encode = torch.nn.ModuleList([
            torch.nn.Linear(ninput if l == 0 else nhidden[l - 1], nhidden[l] if l != nlayers - 1 else noutput) for l in
            range(nlayers)])
        self.BatchNormList = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=nhidden[l] if l != nlayers-1 else noutput) for l in range(nlayers)])
        self.LayerNormList = torch.nn.ModuleList([
            torch.nn.LayerNorm(nhidden[l] if l != nlayers - 1 else noutput) for l in range(nlayers)])

    def forward(self, x):
        # x [B, 220] or [B, 881]
        for l, linear in enumerate(self.encode):
            x = F.relu(linear(x))
            x = self.BatchNormList[l](x)
            x = self.LayerNormList[l](x)
            x = F.dropout(x, self.dropout)
        return x

class DTI_Decoder(nn.Module):
    def __init__(self, Protein_num, Drug_num, Nodefeat_size, nhidden, nlayers, dropout=0.3):
        super(DTI_Decoder, self).__init__()
        self.Protein_num = Protein_num
        self.Drug_num = Drug_num
        self.dropout = dropout
        self.nlayers = nlayers
        self.decode = torch.nn.ModuleList([
            torch.nn.Linear(Nodefeat_size if l == 0 else nhidden[l - 1], nhidden[l] if l != nlayers - 1 else 1) for l in
            range(nlayers)])
        self.BatchNormList = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=nhidden[l]) for l in range(nlayers-1)])

    def forward(self, nodes_features):
        protein_features = nodes_features[:self.Protein_num]
        drug_features = nodes_features[self.Protein_num:]
        protein_features = torch.repeat_interleave(protein_features, repeats=self.Drug_num, dim=0)
        drug_features = drug_features.view(1, self.Drug_num, drug_features.shape[-1])
        drug_features = torch.repeat_interleave(drug_features, repeats=self.Protein_num, dim=0).view(self.Protein_num*self.Drug_num, -1)
        pair_nodes_features = protein_features*drug_features
        
        for l, dti_nn in enumerate(self.decode):
            pair_nodes_features = F.relu(dti_nn(pair_nodes_features))
            if l != self.nlayers - 1:
                pair_nodes_features = self.BatchNormList[l](pair_nodes_features)
            pair_nodes_features = F.dropout(pair_nodes_features, self.dropout)
        output = pair_nodes_features.view(self.Protein_num, self.Drug_num)
        return torch.sigmoid(output)

class DTI_Graph(nn.Module):
    """
    Model for Drug-Protein interaction Graph
    pnn_hyper = [protein_ninput, pnn_nhid, gat_ninput, pnn_nlayers]
    dnn_hyper = [drug_ninput, dnn_nhid, gat_ninput, dnn_nlayers]
    GAT_hyper = [gat_ninput, gat_nhid, gat_noutput, gat_negative_slope, nheads]
    Deco_hyper = [gat_noutput, DTI_nn_nhid]
    """
    def __init__(self, GAT_hyper, PNN_hyper, DNN_hyper, DECO_hyper, Protein_num, Drug_num, dropout):
        super(DTI_Graph, self).__init__()
        self.drug_nn = NN(DNN_hyper[0], DNN_hyper[1], DNN_hyper[2], DNN_hyper[3], dropout)
        self.protein_nn = NN(PNN_hyper[0], PNN_hyper[1], PNN_hyper[2], PNN_hyper[3], dropout)
        self.gat = GAT(GAT_hyper[0], GAT_hyper[1], GAT_hyper[2], dropout, GAT_hyper[3], GAT_hyper[4])
        self.DTI_Decoder = DTI_Decoder(Protein_num, Drug_num, DECO_hyper[0], DECO_hyper[1], DECO_hyper[2], dropout)
        self.Protein_num = Protein_num
        self.Drug_num = Drug_num
        self.BatchNorm = torch.nn.BatchNorm1d(num_features=GAT_hyper[0])
        self.LayerNorm = torch.nn.LayerNorm(GAT_hyper[0])

    def forward(self, Proteins, Drugs, edge_index):

        # Protein and Drug embeding
        emb_drugs = self.drug_nn(Drugs)

        emb_proteins = self.protein_nn(Proteins)
        # emb_proteins nxi, emb_drugs mxi
        Nodes_features = torch.cat((emb_proteins, emb_drugs), 0)
        Nodes_features = self.BatchNorm(Nodes_features)
        Nodes_features = self.LayerNorm(Nodes_features)
        # gat
        Nodes_features = self.gat(Nodes_features, edge_index)
        # Decoder
        DTI_Interaction_Matrix = self.DTI_Decoder(Nodes_features)

        # output
        output = []
        output.append(DTI_Interaction_Matrix)
        return output




