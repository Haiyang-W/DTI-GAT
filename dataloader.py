import json
import torch
import numpy as np
import random
import torch.nn.functional as F
import functools

def cmp_time(a, b):
    a_num = int(a.split('_')[1])
    b_num = int(b.split('_')[1])
    return a_num - b_num

def pad_tensor(vec, pad):
    """
    pad tensor to fixed length

    :parameter
        vec: tensor to pad
        pad: the size to pad to

    :return
        a new tensor padded to 'pad'
    """
    padded = torch.cat([vec, torch.zeros((pad - len(vec),20), dtype=torch.float)], dim=0).data.numpy()
    return padded

def padding_all(vec, max_len):
    """
    vec: [n, len, feat]
    """
    n = vec.shape[0]
    vec_len = vec.shape[1]
    padded = torch.cat([vec, torch.zeros((n,max_len-vec_len,20), dtype=torch.double)], dim=1).data
    
    return padded


def load_info_data(path):
    ori_data = np.load(path)
    protein_tensor = torch.tensor(ori_data['pssm_arr'], dtype =torch.float) # [n_p ,220]
    drug_tensor = torch.tensor(ori_data['drug_arr'], dtype =torch.float) # [n_d, 881]
    protein_num = protein_tensor.shape[0]
    drug_num = drug_tensor.shape[0]
    node_num = protein_num + drug_num
    return protein_tensor, drug_tensor, node_num, protein_num

def load_pre_process(preprocess_path):
    with open(preprocess_path, 'r') as f:
        a = json.load(f)
        adj = torch.FloatTensor(a['adj'])
        dti_inter_mat = torch.FloatTensor(a['dti_inter_mat'])
        train_interact_pos = torch.tensor(a['train_interact_pos'])
        val_interact_pos = torch.tensor(a['val_interact_pos'])
    return adj, dti_inter_mat, train_interact_pos, val_interact_pos







