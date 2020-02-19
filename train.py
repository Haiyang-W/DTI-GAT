import json
import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import DTI_Graph
from dataloader import load_info_data, load_pre_process
from utils import accuracy, precision, recall, specificity, mcc, auc, aupr

###############################################################
# Training settings
parser = argparse.ArgumentParser(description='DTI-GRAPH')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=223, help='Random seed.')
parser.add_argument('--epochs', type=int, default=3000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--model_dir', type=str, default='./enzyme_model_com3',
                    help='model save path')
###############################################################
# Model hyper setting
# Protein_NN
parser.add_argument('--protein_ninput', type=int, default=220,
                    help='protein vector size')
parser.add_argument('--pnn_nlayers', type=int, default=1,
                    help='Protein_nn layers num')
parser.add_argument('--pnn_nhid', type=list, default=[],
                    help='pnn hidden layer dim, like [200,100] for tow hidden layers')
# Drug_NN
parser.add_argument('--drug_ninput', type=int, default=881,
                    help='Drug fingerprint dimension')
parser.add_argument('--dnn_nlayers', type=int, default=1,
                    help='dnn_nlayers num')
parser.add_argument('--dnn_nhid', type=list, default=[],
                    help='dnn hidden layer dim, like [200,100] for tow hidden layers')
# GAT
parser.add_argument('--gat_type', type=str, default='PyG',
                    help="two different type, 'PyG Sparse GAT'(PyG) and 'Dense GAT Self'(Dense-Self)")
parser.add_argument('--gat_ninput', type=int, default=256,
                    help='GAT node feature length, is also the pnn  outpu size and dnn output size')
parser.add_argument('--gat_nhid', type=int, default=256,
                    help='hidden dim of gat')
parser.add_argument('--gat_noutput', type=int, default=256,
                    help='GAT output feature dim and the input dim of Decoder')
parser.add_argument('--gat_nheads', type=int, default=8,
                    help='GAT layers')
parser.add_argument('--gat_negative_slope', type=float, default=0.2,
                    help='GAT LeakyReLU angle of the negative slope.')
# Decoder
parser.add_argument('--DTI_nn_nlayers', type=int, default=2,
                    help='Protein_nn layers num')
parser.add_argument('--DTI_nn_nhid', type=list, default=[256, 256],
                    help='DTI_nn hidden layer dim, like [200,100] for tow hidden layers')
###############################################################
# data
parser.add_argument('--dataset', type=str, default='enzyme',
                    help='dataset name')
parser.add_argument('--common_neighbor', type=int, default=1,
                    help='common neighbor of adj transform, this will determine what preprocessed matrix you use')
parser.add_argument('--sample_num', type=int, default=300,
                    help='different epoch use different sample, the sample num')
parser.add_argument('--data_path', type=str, default='./data',
                    help='dataset root path')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# load data
data_Path = os.path.join(args.data_path, 'mx_'+args.dataset+'.npz')
preprocess_path = os.path.join(args.data_path, 'preprocess', args.dataset+'_com_'+str(args.common_neighbor))
# save dir
if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)
protein_tensor, drug_tensor, node_num, protein_num = load_info_data(data_Path)
# Hyper Setting
pnn_hyper = [args.protein_ninput, args.pnn_nhid, args.gat_ninput, args.pnn_nlayers]
dnn_hyper = [args.drug_ninput, args.dnn_nhid, args.gat_ninput, args.dnn_nlayers]
GAT_hyper = [args.gat_ninput, args.gat_nhid, args.gat_noutput, args.gat_negative_slope, args.gat_nheads]
Deco_hyper = [args.gat_noutput, args.DTI_nn_nhid, args.DTI_nn_nlayers]

def train(epoch, link_dti_id_train, edge_index, edge_weight, train_dti_inter_mat):
    # if use PyG's sparse gcn, you will need the edge_weight
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(protein_tensor, drug_tensor, edge_index)
    row_dti_id = link_dti_id_train.permute(1, 0)[0]
    col_dti_id = link_dti_id_train.permute(1, 0)[1]
    Loss = nn.BCELoss()
    loss_train = Loss(output[0][row_dti_id, col_dti_id], train_dti_inter_mat[row_dti_id, col_dti_id])
    acc_dti_train = accuracy(output[0][row_dti_id, col_dti_id], train_dti_inter_mat[row_dti_id, col_dti_id])
    loss_train.backward()
    optimizer.step()
    print('Epoch {:04d} Train '.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_dti_train: {:.4f}'.format(acc_dti_train),
          'time: {:.4f}s'.format(time.time() - t))

def test(link_dti_id_test, edge_index, edge_weight, test_dti_inter_mat):
    # if use PyG's sparse gcn, you will need the edge_weight
    model.eval()
    row_dti_id = link_dti_id_test.permute(1, 0)[0]
    col_dti_id = link_dti_id_test.permute(1, 0)[1]
    output = model(protein_tensor, drug_tensor, edge_index)
    Loss = nn.BCELoss()
    predicts = output[0][row_dti_id, col_dti_id]
    targets = test_dti_inter_mat[row_dti_id, col_dti_id]
    loss_test = Loss(predicts, targets)
    acc_dti_test = accuracy(output[0][row_dti_id, col_dti_id], test_dti_inter_mat[row_dti_id, col_dti_id])
    return acc_dti_test, loss_test, predicts, targets

# Train model
t_total = time.time()
acc_score = np.zeros(5)
precision_score = np.zeros(5)
recall_score = np.zeros(5)
specificity_score = np.zeros(5)
mcc_score = np.zeros(5)
auc_score = np.zeros(5)
aupr_score = np.zeros(5)
for train_times in range(5):
    model = DTI_Graph(GAT_hyper=GAT_hyper, PNN_hyper=pnn_hyper, DNN_hyper=dnn_hyper, DECO_hyper=Deco_hyper,
                      Protein_num=protein_tensor.shape[0], Drug_num=drug_tensor.shape[0], dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_test = 0
    preprocess_oripath = os.path.join(preprocess_path, '0_'+str(train_times)+'.json')  # ori fold we use test
    adj, ori_dti_inter_mat, ori_train_interact_pos, ori_val_interact_pos = load_pre_process(preprocess_oripath)
    edge_index = torch.nonzero(adj > 0).permute(1, 0)
    edge_weight = adj[np.array(edge_index)]
    if args.cuda:
        model = model.cuda()
        protein_tensor = protein_tensor.cuda()
        drug_tensor = drug_tensor.cuda()
        edge_index = edge_index.cuda()
        # edge_weight = edge_weight.cuda() # if you want to use gcn and so on
        ori_dti_inter_mat = ori_dti_inter_mat.cuda()
        ori_train_interact_pos = ori_train_interact_pos.cuda()
        ori_val_interact_pos = ori_val_interact_pos.cuda()
    save_time_fold = os.path.join(args.model_dir, str(train_times))
    if not os.path.exists(save_time_fold):
        os.mkdir(save_time_fold)
    for epoch in range(args.epochs):
        data_id = epoch % args.sample_num
        print("use sample", data_id)
        preprocess_generate_path = os.path.join(preprocess_path, str(data_id)+'_'+str(train_times)+'.json')
        adj, dti_inter_mat, train_interact_pos, val_interact_pos = load_pre_process(preprocess_generate_path)
        if args.cuda:
            dti_inter_mat = dti_inter_mat.cuda()
            train_interact_pos = train_interact_pos.cuda()
        print('Epoch: {:04d}'.format(epoch + 1), 'Train_times:', train_times)
        train(epoch, train_interact_pos, edge_index, edge_weight, dti_inter_mat)
        test_score, test_loss, predicts, targets = test(ori_val_interact_pos, edge_index, edge_weight, ori_dti_inter_mat)
        if test_score > best_test:
            best_test = test_score
            acc_score[train_times] = best_test
            save_model_path = os.path.join(save_time_fold, preprocess_path.split('/')[-1]+'_times_'+str(train_times)+'_'+
                                     str(round(best_test, 4))+'.pth.tar')
            torch.save(model.state_dict(), save_model_path)
            save_predict_target_path = os.path.join(save_time_fold, preprocess_path.split('/')[-1]+'_times_'+str(train_times)+'_'+
                                     str(round(best_test, 4))+'.txt')
            predict_target = torch.cat((predicts, targets), dim=0).detach().cpu().numpy()
            np.savetxt(save_predict_target_path, predict_target)

            precision_score[train_times] = precision(predicts, targets)
            recall_score[train_times] = recall(predicts, targets)
            specificity_score[train_times] = specificity(predicts, targets)
            mcc_score[train_times] = mcc(predicts, targets)
            auc_score = auc(predicts, targets)
            aupr_score = aupr(predicts, targets)
        print('Epoch: {:04d}'.format(epoch + 1), 'Train_times:', train_times)
        print("*****************test_score {:.4f} best_socre {:.4f}****************".format(test_score, best_test))
        print("All Test Score:", acc_score)
print(args.dataset, " Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print("acc Score:", acc_score)
print("precision Score:", precision_score)
print("recall score", recall_score)
print("specificity score", specificity_score)
print("mcc score", mcc_score)
print("auc socre", auc_score)
print("aupr score", aupr_score)
print("Best Ave Test: {:.4f}".format(np.mean(acc_score)))