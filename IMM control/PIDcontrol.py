import torch
import os
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.signal import lti
import numpy as np
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DeltaPID:
    """增量式PID算法实现"""

    def __init__(self, dt, p, i, d) -> None:
        self.dt = dt  # 循环时间间隔
        self.k_p = p  # 比例系数
        self.k_i = i  # 积分系数
        self.k_d = d  # 微分系数
        self._pre_error = 0  # t-1 时刻误差值
        self._pre_pre_error = 0  # t-2 时刻误差值

    def calcalate(self, aim, cur_u, cur_y):

        error = aim - cur_y
        p_change = self.k_p * (error - self._pre_error)
        i_change = self.k_i * error
        d_change = self.k_d * (error - 2 * self._pre_error + self._pre_pre_error)
        delta_output = p_change + i_change + d_change  # 本次增量
        cur_u += delta_output  # 计算当前位置

        self._pre_error = error
        self._pre_pre_error = self._pre_error

        return cur_u

Ts = 0.005

n_x = 4
n_u = 1
n_y = 1

NTs = 200

A = np.array([[0.7805, 1, 0, 0], [-0.1538, 0, 1, 0], [0.0121, 0, 0, 1], [0, 0, 0, 0]])
B = np.array([[0.3941], [-0.0085], [-0.0073], [0.0011]])
D = np.array([[0.8805], [-0.0538], [0.1121], [0.1]])
H = np.array([1, 0, 0, 0]).reshape(1, 4)

noice_ratio = 0.5

Q_y = 1 * np.eye(n_y)
Q_u = 1 * np.eye(n_u)
Q_y0 = 0 * np.eye(n_y)
Q_u0 = 0 * np.eye(n_u)
lb_u = 0 * np.ones(n_u)
ub_u = 20 * np.ones(n_u)
lb_y = 0 * np.ones(n_y)
ub_y = 20 * np.ones(n_y)

x0 = np.zeros(n_x)
u0 = np.zeros(n_u)
y0 = np.zeros(n_y)

x_out = np.zeros((n_x, NTs))
y_out = np.zeros((n_y, NTs))
u_out = np.zeros((n_u, NTs))

aim = 10

noise_ = torch.normal(mean=torch.full((1, NTs), 0.0), std=torch.full((1, NTs), 0.1))

kp = 0.3
ki = 0.18
kd = 0.01
pid = DeltaPID(Ts, kp, ki, kd)

cur_val = 0

for i in range(NTs):

    noise = noise_[0, i]

    x_out[:, i] = x0[:]
    y_out[:, i] = y0[:]
    u_out[:, i] = u0[:]

    u0[:] = pid.calcalate(aim, u0, y0)

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
plt.plot(range(NTs), u_out, 'r', label="PID-u")
plt.legend()
plt.show()
plt.plot(range(NTs), y_out, 'r', label="PID-y")
plt.legend()
plt.show()
