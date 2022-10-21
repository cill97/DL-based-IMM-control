import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from data_preprocessing import read_file, min_max, un_min_max, min_max_remake
from train_save import load_model

Ts = 0.005

n_x = 4
n_u = 1
n_y = 1

NTs = 200

A = np.array([[0.7805, 1, 0, 0], [-0.1538, 0, 1, 0], [0.0121, 0, 0, 1], [0, 0, 0, 0]])
B = np.array([[0.3941], [-0.0085], [-0.0073], [0.0011]])
D = np.array([[0.8805], [-0.0538], [0.1121], [0.1]])
H = np.array([1, 0, 0, 0]).reshape(1, 4)

noice_ratio = 0

x0 = np.zeros(n_x)
u0 = np.zeros(n_u)
y0 = np.zeros(n_y)

x_out = np.zeros((n_x, NTs))
y_out = np.zeros((n_y, NTs))
u_out = np.zeros((n_u, NTs))

noise_ = torch.normal(mean=torch.full((1, NTs), 0.0), std=torch.full((1, NTs), 0.1))

in_file = './train_data/inject_input1.npy'
out_file = './train_data/inject_output1.npy'
input_data = read_file(in_file)
output_data = read_file(out_file)
min_input, max_input, input_data = min_max_remake(input_data, "std")
min_output, max_output, output_data = min_max_remake(output_data, "std", min_input[6], max_input[6])

net = load_model(enc_hid_dim=8, dec_hid_dim=8, input_dim=6, output_dim=3, emb_dim=7, final_dim=1, acti='ReLU')
net.load_state_dict(torch.load("./save_weight/GRU_MPC+mode_min_max_or_not+min_max.pkl"))

aim = np.zeros(NTs)

for i in range(NTs):
    if 0 <= i < 16:
        aim[i] = 0.5 * i
    elif 16 <= i < 40:
        aim[i] = 8
    elif 40 <= i < 56:
        aim[i] = aim[i-1] - 0.1875
    elif 56 <= i < 80:
        aim[i] = 5
    elif 80 <= i < 120:
        aim[i] = aim[i-1] + 0.05
    elif 120 <= i < 160:
        aim[i] = 7
    elif 160 <= i < 185:
        aim[i] = aim[i-1] - 0.2
    elif 185 <= i < 200:
        aim[i] = 2

net_input = np.zeros((8, 7))
sos = -1 * np.ones(7)
newdata = np.zeros((1, 7))
trg = torch.tensor(-1 * np.ones((1, 4, 1)), dtype=torch.float32)
net_input = min_max(net_input, "std", min_input, max_input)
net_input[0, :] = sos[:]
net_input[7, :] = sos[:]
for i in range(NTs):

    noise = noise_[0, i]

    x_out[:, i] = x0[:]
    y_out[:, i] = y0[:]
    u_out[:, i] = u0[:]

    net_input[1:6, :] = net_input[2:7, :]

    newdata[0, 0:4] = x0[:]
    newdata[0, 4] = y0[:]
    newdata[0, 5] = aim[i]
    newdata[0, 6] = u0[:]
    newdata = min_max(newdata, "std", min_input, max_input)
    net_input[6, :] = newdata[0, :]

    u0[:] = net(torch.tensor(net_input.reshape(1, 8, 7), dtype=torch.float32), trg).detach().numpy()

    u0 = un_min_max(u0, min_output, max_output, "un_std")

    u1 = u0.reshape(1, 1)
    # predicted
    x0_hat = x0.reshape(4, 1)
    x1_hat = A.dot(x0_hat) + B.dot(u1)
    # updated
    x0 = A.dot(x0_hat) + B.dot(u1) + D.dot(noise).reshape(4, 1)
    y0 = H.dot(x0)

    x0 = x0.reshape(4)
    y0 = y0.reshape(1)

y_out = y_out.reshape(NTs)  # 601
u_out = u_out.reshape(NTs)  # 601
plt.plot(range(NTs), u_out, 'r', label="DL-u")
plt.legend()
plt.show()
plt.plot(range(NTs), y_out, 'r', label="DL-y")
plt.legend()
plt.show()
