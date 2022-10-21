import torch
import numpy as np
import math
import random
from torch import nn
import os
import datetime
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from data_preprocessing import read_file, in_data_prepare, dataloader_create, min_max, un_min_max, min_max_remake
from seq2seq import Attention, Encoder, Decoder, Seq2Seq
from train_save import train, load_model, repeat_train

if __name__ == "__main__":
    in_file = './train_data/inject_input1.npy'
    out_file = './train_data/inject_output1.npy'
    input_data = read_file(in_file)
    output_data = read_file(out_file)
    params = {}
    # 编码器输入数据时间步长度 输入数据预测维度 [2, 4, 6, 8, 10, 12, 16, 20]
    params['input_dim'] = [2, 4, 6, 8, 10, 12, 16, 20]
    # 解码器输入数据时间步长度 输入数据预测维度 [2, 4, 6, 8, 10, 12, 16, 20]
    params['output_dim'] = [2, 4, 6, 8, 10, 12, 16, 20]
    # 编码器输入数据每一时间步长度 [x1,x2,x3,x4,y,y_aim,u]
    params['emb_dim'] = 7
    # 隐藏状态维度 [5, 10, 15, 20, 30]
    params['hid_dim'] = [1, 2, 3, 4, 5, 7, 8, 10, 15, 20]
    # 最终输出维度, u
    params['final_dim'] = 1
    # 验证集比例
    params['percent'] = [0.1]
    # 训练次数
    params['epochs'] = 100
    # 显示输出
    params['log_step'] = 200
    # 激活函数 'tanh', 'ReLU', 'Sigmoid'
    params['activation'] = ['ReLU', 'tanh', 'Sigmoid']
    params['lr'] = 0.001
    # params['weight_decay'] = 1e-3
    # 训练模式 'activation', 'in_out_dim', 'hidden_dim', 'min_max_or_not', 'percent'
    params['mode'] = ['min_max_or_not', 'hidden_dim', 'in_out_dim'] #, 'hidden_dim', 'in_out_dim'
    # 数据处理与否 'not_min_max', 'min_max'
    params['data_process'] = ['min_max']

    # 标准化
    for mode in params['data_process']:
        if mode == 'min_max':
            min_input, max_input, input_data = min_max_remake(input_data, "std")
            min_output, max_output, output_data = min_max_remake(output_data, "std", min_input[6], max_input[6])

        # 加载训练
        repeat_train(params, input_data, output_data, mode)
