import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

#加载矩阵
def load_adj(dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)

    #sp.load_npz 是 scipy.sparse 的函数，用于加载保存为 .npz 格式的稀疏矩阵。在这里，加载的是邻接矩阵 adj.npz
    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    #将加载的矩阵转换为CompressedSparseColumn(CSC)格式。这种格式对列切片操作更高效，适用于某些类型的矩阵运算。
    adj = adj.tocsc()
    
    if dataset_name == 'metr-la':
        n_vertex = 207
    elif dataset_name == 'pems-bay':
        n_vertex = 325
    elif dataset_name == 'pemsd7-m':
        n_vertex = 228

    #n_vertex是顶点的数量
    return adj, n_vertex


#返回数据集
def load_data(dataset_name, len_train, len_val):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))

    train = vel[: len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:]
    return train, val, test

#分割数据集
def data_transform(data, n_his, n_pred, device):
    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred

    x = np.zeros([num, n_his, n_vertex])
    y = np.zeros([num, n_vertex])  # 修改为[num, 1, n_vertex]

    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :] = data[head: tail]
        y[i, :] = data[tail + n_pred - 1]  # 确保 y 的形状为[num, 1, n_vertex]

    # 创建张量时设置 requires_grad=True
    return (torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)), (torch.tensor(y, dtype=torch.float32, requires_grad=True).to(device))

