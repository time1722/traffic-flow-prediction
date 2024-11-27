import logging
import os
import gc
import argparse
import math
import random
import warnings
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from script import dataloader, utility, earlystopping, opt
from model import models

#import nni

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def split_batches(tensor):
    """
    分割形状为 [batchsize, seq_len, feature] 的张量的每个 batch。

    :param tensor: 输入的张量，形状为 [batchsize, seq_len, feature]
    :return: 一个包含每个 batch 的列表
    """
    batchsize, seq_len, feature = tensor.shape

    batches = []
    # 遍历 batch 维度，分割出每个 batch
    for batch_idx in range(batchsize):
        # 获取当前 batch 的张量，形状为 [seq_len, feature]
        current_batch = tensor[batch_idx, :, :]
        batches.append(current_batch)

    return batches
def set_env(seed):
    # Set available CUDA devices
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='metr-la', choices=['metr-la', 'pems-bay', 'pemsd7-m'])
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=9, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--graph_conv_type', type=str, default='InsiderClassifier', choices=['InsiderClassifier','cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.001, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000, help='epochs, default as 1000')
    parser.add_argument('--opt', type=str, default='nadamw', choices=['adamw', 'nadamw', 'lion'], help='optimizer, default as nadamw')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
        torch.cuda.empty_cache() # Clean cache
    else:
        device = torch.device('cpu')
        gc.collect() # Clean cache
    
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st0_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    
    return args, device, blocks

def data_preparate(args, device):    
    adj, n_vertex = dataloader.load_adj(args.dataset)
    #计算和获取图的谱特征
    gso = utility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    #shape[0]是获取DataFrame的行数，表示数据集中有多少条记录
    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 70: 15: 15
    val_and_test_rate = 0.15

    #获取验证，测试，训练的行数，math.floor是向下取整
    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    #数据标准化
    train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
    x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, device)
    x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, device)

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, zscore, train_iter, val_iter, test_iter

def prepare_model(args, blocks, n_vertex):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(delta=0.0, 
                                     patience=args.patience, 
                                     verbose=True, 
                                     path="InsiderClassifier" + args.dataset + ".pt")

    if args.graph_conv_type == 'InsiderClassifier':
        model = models.InsiderClassifier(
        num_features=0,  # 假设无额外数值特征
        cat_features=['hist_activity'],
        seq_features=['hist_activity'],
        cat_nums={'hist_activity': 10},
        cat_embedding_size=16,
        seq_embedding_size=228,
        lstm_hidden_size=228,
        dropout=0.1,
        reduction=False,
        use_attention=False,
        LayerNorm=True,
        encoder_type='lstm',
        num_lstm_layers=2,
        mlp_hidden_layers=[],
        pooling_mode='origin',
        epsilon=0,
        num_pers=4,
        graph_metric_type='weighted_cosine',
        topk=12,
        num_class=228,
        add_graph_regularization=False,
        gnn='GCN',
        embedding_hook=False
    )

    if args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "nadamw":
        optimizer = optim.NAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, decoupled_weight_decay=True)
    elif args.opt == "lion":
        optimizer = opt.Lion(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    else:
        raise ValueError(f'ERROR: The {args.opt} optimizer is undefined.')

    #学习率调度器 StepLR，args.step_size表示训练几次更新一次学习率，args.gamma衰减因子，表示每次更新时学习率都乘上衰减因子
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler

def train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter,zscore,test_iter):
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            if y.shape[0] != args.batch_size:
                print(f"Skipping batch with size {y.shape[0]} (expected {args.batch_size})")
                continue
            optimizer.zero_grad()
            x1 = {
                'hist_activity': x,  # 每个样本的历史活动序列
                'max_len': torch.tensor([12] * 32, dtype=torch.float32)  # 每个样本的序列实际长度
            }
            y_pred = model(x1)
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = val(model, val_iter)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))
        test(zscore, loss, model, test_iter, args)

        es(val_loss, model)
        if es.early_stop:
            print("Early stopping")
            break

#用于在特定函数或代码块中禁用梯度计算
@torch.no_grad()
def val(model, val_iter):
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        if y.shape[0] != args.batch_size:
            print(f"Skipping batch with size {y.shape[0]} (expected {args.batch_size})")
            continue
        x1 = {
            'hist_activity': x,  # 每个样本的历史活动序列
            'max_len': torch.tensor([12] * 32, dtype=torch.float32)  # 每个样本的序列实际长度
        }
        y_pred = model(x1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)

@torch.no_grad() 
def test(zscore, loss, model, test_iter, args):
    model.eval()

    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, zscore)
    print(f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

if __name__ == "__main__":
    # Logging
    #logger = logging.getLogger('stgcn')
    #logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


    args, device, blocks = get_parameters()
    n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)
    train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter,zscore,test_iter)
    test(zscore, loss, model, test_iter, args)
