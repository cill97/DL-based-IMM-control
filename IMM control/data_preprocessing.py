import numpy as np
import math
import random
import torch
from torch.utils.data import DataLoader, TensorDataset


def read_file(filename):
    data = np.load(filename)
    return data.astype(np.float32)


def dataloader_create(enArr, deArr, preArr, Percent):
    rows = enArr.shape[0]
    validationSplitPercent = Percent  # 验证集比例
    numValidationDataRows = math.floor(validationSplitPercent * rows)

    validId = random.sample(range(rows), numValidationDataRows)

    val_enc = enArr[validId, :, :]
    val_dec = deArr[validId, :, :]
    val_pre = preArr[validId, :]

    # 获取训练集数据并打乱
    trainId = set(range(rows)).difference(set(validId))

    tra_enc = enArr[list(trainId), :, :]
    tra_dec = deArr[list(trainId), :]
    tra_pre = preArr[list(trainId), :]

    dl_train = DataLoader(
        TensorDataset(torch.tensor(tra_enc, dtype=torch.float32), torch.tensor(tra_dec, dtype=torch.float32),
                      torch.tensor(tra_pre, dtype=torch.float32)), shuffle=True, batch_size=128)
    dl_val = DataLoader(
        TensorDataset(torch.tensor(val_enc, dtype=torch.float32), torch.tensor(val_dec, dtype=torch.float32),
                      torch.tensor(val_pre, dtype=torch.float32)), shuffle=False, batch_size=128)
    return dl_train, dl_val


def in_data_prepare(encoder_data_dim, decoder_data_dim, indata, outdata):
    # data_dim 时间步维度 data 数据集
    data_set = indata
    pre_set = outdata
    pre_size = outdata.shape[2]
    dataset_step = data_set.shape[1]  # 200
    dataset_arr = data_set.shape[2]  # 7
    enc_cut_size = encoder_data_dim + 2  # 10 + 'SOS'
    dec_cut_size = decoder_data_dim + 1  # 8 + 'SOS'
    dataset_size = data_set.shape[0]  # 360
    # 预留首位为开始标志符，定义开始标志符为全-1，一二项时间步不足用0补齐
    encoderArr = -1 * np.ones((dataset_size, dataset_step - decoder_data_dim, enc_cut_size, dataset_arr))
    decoderArr = -1 * np.ones((dataset_size, dataset_step - decoder_data_dim, dec_cut_size, dataset_arr))
    decoderArr2 = -1 * np.ones((dataset_size, dataset_step - decoder_data_dim, dec_cut_size, pre_size))
    pre_label = np.zeros((dataset_size, dataset_step - decoder_data_dim, pre_size))
    for i in range(dataset_size):
        addzeros = np.zeros((encoder_data_dim - 1, dataset_arr))
        addata = data_set[i, :, :]
        datain = np.vstack((addzeros, addata))
        datapre = pre_set[i, :, :]
        inArr, deArr, preArr, deArr2 = in_out_creat(datain, datapre, encoder_data_dim, decoder_data_dim, dataset_arr, pre_size)
        encoderArr[i, :, 1:enc_cut_size-1, :] = inArr[:, :, :]
        decoderArr[i, :, 1:dec_cut_size, :] = deArr[:, :, :]
        decoderArr2[i, :, 1:dec_cut_size, :] = deArr2[:, :, :]
        pre_label[i, :, :] = preArr[:, :]
    encoderArr = encoderArr.reshape(dataset_size * (dataset_step - decoder_data_dim), enc_cut_size, dataset_arr)
    decoderArr = decoderArr.reshape(dataset_size * (dataset_step - decoder_data_dim), dec_cut_size, dataset_arr)
    decoderArr2 = decoderArr2.reshape(dataset_size * (dataset_step - decoder_data_dim), dec_cut_size, pre_size)
    pre_label = pre_label.reshape(dataset_size * (dataset_step - decoder_data_dim), pre_size)
    return encoderArr, decoderArr, pre_label, decoderArr2


def in_out_creat(Arr, Arrl, windowSize, preStep, dataset_arr, pre_size):
    step = Arr.shape[0] - preStep - windowSize + 1
    inpArr = np.zeros((step, windowSize, dataset_arr))
    outArr = np.zeros((step, preStep, dataset_arr))
    outArr2 = np.zeros((step, preStep))
    preArr = np.zeros((step, pre_size))
    for i in range(step):
        inpArr[i, :, :] = Arr[i:i + windowSize, :]
        outArr[i, :, :] = Arr[i + windowSize: i + windowSize + preStep, :]
        outArr2[i, :] = Arr[i + windowSize: i + windowSize + preStep, 6]
        preArr[i, :] = Arrl[i, :]
    return inpArr, outArr, preArr, outArr2.reshape(step, preStep, pre_size)


def min_max(data, mode="std", minn=None, maxx=None):
    feature_range = [0, 1]
    if (minn is None) * (maxx is None):
        min = data.min(axis=0)
        max = data.max(axis=0)
    else:
        min = minn
        max = maxx
    x_std = (data - min) / (max - min)
    if mode == "std":  # 标准化
        return x_std
    elif mode == "scaled":  # 归一化
        x_scaled = x_std * (feature_range[1] - feature_range[0]) + feature_range[0]
        return x_scaled
    elif mode == "min_max":  # 输出最大最小值
        return data.min(axis=0), data.max(axis=0)


def un_min_max(data, min, max, mode="un_std"):
    feature_range = [0, 1]
    if mode == "un_std":  # 逆标准化
        x_unstd = data * (max - min) + min
        return x_unstd
    elif mode == "un_scaled":  # 逆归一化
        x_unscaled = (data - feature_range[0]) / (feature_range[1] - feature_range[0])
        x_unstd = x_unscaled * (max - min) + min
        return x_unstd


def min_max_remake(data, mode, minn=None, maxx=None):
    x = data.shape[0]
    y = data.shape[1]
    z = data.shape[2]
    data_hat = data.reshape(x * y, z)
    if (minn is None) * (maxx is None):
        min, max = min_max(data_hat, "min_max")
        data_hat = min_max(data_hat, mode)
    else:
        min = minn
        max = maxx
        data_hat = min_max(data_hat, mode, min, max)
    data_hat = data_hat.reshape(x, y, z)
    return min, max, data_hat


def un_min_max_remake(data, mode, min, max):
    x = data.shape[0]
    y = data.shape[1]
    z = data.shape[2]
    data_hat = data.reshape(x * y, z)
    data_hat = un_min_max(data_hat, min, max, mode)
    data_hat = data_hat.reshape(x, y, z)
    return data_hat
