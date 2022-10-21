import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):  # 10, 7, 20, 20
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, acti='tanh'):
        super().__init__()
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        if acti == 'tanh':
            self.acti = nn.Tanh()
        elif acti == 'ReLU':
            self.acti = nn.ReLU()
        elif acti == 'Sigmoid':
            self.acti = nn.Sigmoid()

    def forward(self, src):
        '''
        src = [src_len, batch_size, 7]
        '''
        # src = [src_len, batch_size, emb_dim]
        enc_output, enc_hidden = self.rnn(src)
        s = self.acti(self.fc(torch.cat((enc_hidden[-2, :, :], enc_hidden[-1, :, :]), dim=1)))
        return enc_output, s


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        batch_size = enc_output.shape[1]
        src_len = enc_output.shape[0]

        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        enc_output = enc_output.transpose(0, 1)

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, attention, acti='tanh'):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.attention = attention
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, emb_dim)
        if acti == 'tanh':
            self.acti = nn.Tanh()
        elif acti == 'ReLU':
            self.acti = nn.ReLU()
        elif acti == 'Sigmoid':
            self.acti = nn.Sigmoid()

    def forward(self, dec_input, s, enc_output):
        # dec_input = [batch_size, 7]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        dec_input = dec_input.unsqueeze(1)  # dec_input = [batch_size, 1, 1]

        embedded = dec_input.transpose(0, 1)  # embedded = [1, batch_size, 1]

        # a = [batch_size, 1, src_len]
        a = self.attention(s, enc_output).unsqueeze(1)

        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        enc_output = enc_output.transpose(0, 1)

        # c = [1, batch_size, enc_hid_dim * 2]
        c = torch.bmm(a, enc_output).transpose(0, 1)

        # rnn_input = [1, batch_size, (enc_hid_dim * 2) + emb_dim]
        rnn_input = torch.cat((embedded, c), dim=2)

        # dec_output = [src_len(=1), batch_size, dec_hid_dim]
        # dec_hidden = [n_layers * num_directions, batch_size, dec_hid_dim]
        s = s.unsqueeze(0)
        dec_output, dec_hidden = self.rnn(rnn_input, s)
        # 1,100,17  1,100,5
        # embedded = [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # c = [batch_size, enc_hid_dim * 2]
        embedded = embedded.squeeze(0)
        dec_output = dec_output.squeeze(0)
        c = c.squeeze(0)

        # pred = [batch_size, output_dim]
        pred = self.acti(self.fc_out(torch.cat((dec_output, c, embedded), dim=1)))

        return pred, dec_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, output_dimm):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc = nn.Linear(self.decoder.output_dim, output_dimm)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src_len, batch_size, 7]
        # trg = [trg_len, batch_size, 7]
        src = src.transpose(0, 1)
        trg = trg.transpose(0, 1)

        batch_size = src.shape[1]
        output_len = trg.shape[0]  # trg.shape[0] #修改为输出gru预测长度 10
        output_size = self.decoder.emb_dim  # self.decoder.output_dim 7
        outputs = torch.zeros(output_len, batch_size, output_size)
        enc_output, s = self.encoder(src)
        # first input to the decoder is the <sos> tokens
        dec_input = trg[0, :, :]  # 开始标志位
        # trg_len = 7
        for t in range(1, output_len):
            dec_output, s = self.decoder(dec_input, s, enc_output)

            outputs[t] = dec_output
            # teacher_force = random.random() < teacher_forcing_ratio
            # dec_input = trg[t, :, :] if teacher_force else dec_output
            dec_input = dec_output

        outputs = outputs.transpose(0, 1)
        outputs = outputs.squeeze(2)
        outputs = self.fc(outputs)
        return outputs
