import torch
import numpy as np
from torch import nn
import datetime
import pandas as pd
from seq2seq import Attention, Encoder, Decoder, Seq2Seq
from data_preprocessing import in_data_prepare, dataloader_create


def load_model(enc_hid_dim, dec_hid_dim, input_dim, output_dim, emb_dim, final_dim, acti):
    attn = Attention(enc_hid_dim, dec_hid_dim)
    enc = Encoder(input_dim + 2, emb_dim, enc_hid_dim, dec_hid_dim,
                  acti)  # input_dim, emb_dim, enc_hid_dim, dec_hid_dim, acti='tanh'
    dec = Decoder(output_dim + 1, 1, enc_hid_dim, dec_hid_dim, attn,
                  acti)  # output_dim, emb_dim, enc_hid_dim, dec_hid_dim, attention
    net = Seq2Seq(enc, dec, final_dim)  # encoder, decoder, output_dimm
    return net


def repeat_train(params, input_data, output_data, data):
    for mode in params['mode']:
        # 'activation', 'in_out_dim', 'enc_dec_hidden_dim', 'min_max_or_not'
        if mode == 'activation':
            for activation in params['activation']:
                # 分割数据集
                encoderArr, decoderArr, pre_label, decoderArr2 = in_data_prepare(8, 5, input_data, output_data)
                dl_train, dl_val = dataloader_create(encoderArr, decoderArr2, pre_label, 0.1)
                net = load_model(enc_hid_dim=5,
                                 dec_hid_dim=5,
                                 input_dim=8,
                                 output_dim=5,
                                 emb_dim=7,
                                 final_dim=1,
                                 acti=activation)
                optimizer = torch.optim.Adam(params=net.parameters(), lr=params['lr'])
                loss_func = nn.MSELoss()

                train(model=net, traindata_loader=dl_train, validation_loader=dl_val, loss_function=loss_func,
                      optimizer=optimizer, acti=activation, mode=mode, epochs=params['epochs'],
                      log_step=params['log_step'])

        elif mode == 'in_out_dim':
            for in_dim in params['input_dim']:
                for out_dim in params['output_dim']:
                    encoderArr, decoderArr, pre_label, decoderArr2 = in_data_prepare(in_dim, out_dim, input_data,
                                                                                     output_data)
                    dl_train, dl_val = dataloader_create(encoderArr, decoderArr2, pre_label, 0.1)
                    net = load_model(enc_hid_dim=5,
                                     dec_hid_dim=5,
                                     input_dim=in_dim,
                                     output_dim=out_dim,
                                     emb_dim=7,
                                     final_dim=1,
                                     acti='ReLU')
                    optimizer = torch.optim.Adam(params=net.parameters(), lr=params['lr'])
                    loss_func = nn.MSELoss()

                    train(model=net, traindata_loader=dl_train, validation_loader=dl_val, loss_function=loss_func,
                          optimizer=optimizer, input_dim=in_dim, output_dim=out_dim, mode=mode, epochs=params['epochs'],
                          log_step=params['log_step'])

        elif mode == 'hidden_dim':
            for hid in params['hid_dim']:
                encoderArr, decoderArr, pre_label, decoderArr2 = in_data_prepare(8, 5, input_data, output_data)
                dl_train, dl_val = dataloader_create(encoderArr, decoderArr2, pre_label, 0.1)
                net = load_model(enc_hid_dim=hid,
                                 dec_hid_dim=hid,
                                 input_dim=8,
                                 output_dim=5,
                                 emb_dim=7,
                                 final_dim=1,
                                 acti='ReLU')
                optimizer = torch.optim.Adam(params=net.parameters(), lr=params['lr'])
                loss_func = nn.MSELoss()
                # model, traindata_loader, validation_loader, loss_function, optimizer, hid_dim=10, input_dim=10, output_dim=8,emb_dim=7, final_dim=1, acti='tanh', epochs=100, log_step=1000, mode='min_max_or_not'
                train(model=net, traindata_loader=dl_train, validation_loader=dl_val, loss_function=loss_func,
                      optimizer=optimizer, hid_dim=hid, mode=mode, epochs=params['epochs'], log_step=params['log_step'])

        elif mode == 'min_max_or_not':
            encoderArr, decoderArr, pre_label, decoderArr2 = in_data_prepare(6, 3, input_data, output_data)
            dl_train, dl_val = dataloader_create(encoderArr, decoderArr2, pre_label, 0.1)
            net = load_model(enc_hid_dim=8,
                             dec_hid_dim=8,
                             input_dim=6,
                             output_dim=3,
                             emb_dim=7,
                             final_dim=1,
                             acti='ReLU')
            optimizer = torch.optim.Adam(params=net.parameters(), lr=params['lr'])
            loss_func = nn.MSELoss()
            # model, traindata_loader, validation_loader, loss_function, optimizer, hid_dim=10, input_dim=10, output_dim=8,emb_dim=7, final_dim=1, acti='tanh', epochs=100, log_step=1000, mode='min_max_or_not'
            train(model=net, traindata_loader=dl_train, validation_loader=dl_val, loss_function=loss_func,
                  optimizer=optimizer, mode=mode, da=data, epochs=params['epochs'], log_step=params['log_step'])
        elif mode == 'percent':
            encoderArr, decoderArr, pre_label, decoderArr2 = in_data_prepare(8, 5, input_data, output_data)
            for pern in params['percent']:
                dl_train, dl_val = dataloader_create(encoderArr, decoderArr2, pre_label, pern)
                net = load_model(enc_hid_dim=5,
                                 dec_hid_dim=5,
                                 input_dim=8,
                                 output_dim=5,
                                 emb_dim=7,
                                 final_dim=1,
                                 acti='ReLU')
                optimizer = torch.optim.Adam(params=net.parameters(), lr=params['lr'])
                loss_func = nn.MSELoss()
                # model, traindata_loader, validation_loader, loss_function, optimizer, hid_dim=10, input_dim=10, output_dim=8,emb_dim=7, final_dim=1, acti='tanh', epochs=100, log_step=1000, mode='min_max_or_not'
                train(model=net, traindata_loader=dl_train, validation_loader=dl_val, loss_function=loss_func,
                      optimizer=optimizer, mode=mode, da=data, epochs=params['epochs'], log_step=params['log_step'],
                      percent=pern)


def train(model, traindata_loader, validation_loader, loss_function, optimizer, hid_dim=10, input_dim=10, output_dim=8,
          emb_dim=7, final_dim=1, acti='tanh', epochs=100, log_step=100, mode='min_max_or_not', da='min_max',
          percent=0.1):
    print("Start Training...")
    dfhistory = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("==========" * 8 + "%s" % nowtime)
    epochs = epochs
    log_step = log_step

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        step = 1

        for step, (features, trg, labels) in enumerate(traindata_loader, 1):

            optimizer.zero_grad()
            predictions = model(features, trg)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            if step % log_step == 0:
                print(("[step = %d] loss:%f") % (step, loss_sum / step))

        model.eval()
        val_loss_sum = 0.0
        val_step = 1
        for val_step, (features, trg, labels) in enumerate(validation_loader, 1):
            with torch.no_grad():
                predictions = model(features, trg)
                val_loss = loss_function(predictions, labels)
                val_loss_sum += val_loss.item()

        info = (epoch, loss_sum / step, val_loss_sum / val_step)
        dfhistory.loc[epoch - 1] = info

        print("\nEPOCH = %d, loss = %f, val_loss = %f" % info)

        if epoch == 1:
            val_loss_last = val_loss_sum / val_step
            loss_last1 = loss_sum / step

        if val_loss_last > (val_loss_sum / val_step) and loss_last1 > (loss_sum / step):  # 保存模型
            # 训练模式 'activation', 'in_out_dim', 'hidden_dim', 'min_max_or_not'
            if mode == 'activation':
                torch.save(model.state_dict(),
                           './save_weight/GRU_MPC+mode_' + mode + '+' + acti + '.pkl')
            elif mode == 'in_out_dim':
                torch.save(model.state_dict(),
                           './save_weight/GRU_MPC+mode_' + mode + '+inputdim%d_outputdim%d.pkl' % (
                               input_dim, output_dim))
            elif mode == 'hidden_dim':
                torch.save(model.state_dict(),
                           './save_weight/GRU_MPC+mode_' + mode + '+hiddendim%d.pkl' % (hid_dim))
            elif mode == 'min_max_or_not':
                torch.save(model.state_dict(),
                           './save_weight/GRU_MPC+mode_' + mode + '+' + da + '.pkl')
            elif mode == 'percent':
                torch.save(model.state_dict(),
                           './save_weight/GRU_MPC+mode_' + mode + '+%f.pkl' % percent)
            val_loss_last = val_loss_sum / val_step
            loss_last1 = loss_sum / step
            print("\nsave model, loss = %f, val_loss = %f" % (loss_last1, val_loss_last))
        else:
            print("\nbest model, loss = %f, val_loss = %f" % (loss_last1, val_loss_last))

        print("mode=" + mode)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "==========" * 8 + "%s" % nowtime)

    if mode == 'activation':
        dfhistory.to_csv("./train_loss/seq2seq_train_data+mode_" + mode + "+" + acti + ".csv")
    elif mode == 'in_out_dim':
        dfhistory.to_csv(
            "./train_loss/seq2seq_train_data+mode_" + mode + "+inputdim%d_outputdim%d.csv" % (input_dim, output_dim))
    elif mode == 'hidden_dim':
        torch.save(model.state_dict(),
                   './save_weight/GRU_MPC+mode_' + mode + '+hiddendim%d.pkl' % hid_dim)
        dfhistory.to_csv("./train_loss/seq2seq_train_data+mode_" + mode + "+hiddendim%d.csv" % hid_dim)
    elif mode == 'min_max_or_not':
        dfhistory.to_csv("./train_loss/seq2seq_train_data+mode_" + mode + "+" + da + ".csv")
    elif mode == 'percent':
        dfhistory.to_csv("./train_loss/seq2seq_train_data+mode_" + mode + "+%f.csv" % percent)
    print('Finished Training')
