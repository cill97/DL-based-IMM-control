#!/usr/bin/env python
# coding=utf-8

import torch.nn.functional as F
import torch
from torch import nn
import os
import numpy as np
import scipy.linalg as splinalg
import cvxpy as cp
import matplotlib.pyplot as plt
import time
from scipy.signal import lti
import numpy as np
from scipy.signal import lsim
import random
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

A1 = np.array([[-2029, -2544622, -1.9213 * (10 ** 9), -2.0412 * (10 ** 11)], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
B1 = np.array([[1], [0], [0], [0]])
C1 = np.array([[0, 0, 0, 2.144 * (10 ** 11)]])
D1 = 0
system = lti(A1, B1, C1, D1)

Ts = 0.005

n_x = 4
n_u = 1
n_y = 1

NTs = 200
road = 360
A = np.array([[0.7805, 1, 0, 0], [-0.1538, 0, 1, 0], [0.0121, 0, 0, 1], [0, 0, 0, 0]])
B = np.array([[0.3941], [-0.0085], [-0.0073], [0.0011]])
D = np.array([[0.8805], [-0.0538], [0.1121], [0.1]])
H = np.array([1, 0, 0, 0]).reshape(1, 4)

noice_ratio = 0.5

data1 = np.zeros((road, NTs, 7))
data2 = np.zeros((road, NTs, 1))

for aaim in range(road):
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

    aim = random.uniform(0, 20)
    h = 30

    STPP = random.randint(20, 30)
    STPP1 = random.randint(70, 80)
    STPP2 = random.randint(120, 130)
    STPP3 = random.randint(170, 180)

    ran = random.uniform(0.1, 0.5)
    noise_ = torch.normal(mean=torch.full((1, NTs), 0.0), std=torch.full((1, NTs), ran))
    print(aaim)
    print(ran)

    tags = random.random() < noice_ratio
    print(tags)
    for i in range(NTs):
        noise = noise_[0, i] if tags else 0

        if i == STPP or i == STPP1 or i == STPP2 or i == STPP3:
            yi = random.uniform(-ub_y/2, ub_y/2)
            if aim + yi >= 0:
                aim = aim + yi

        ysp = aim * np.ones(n_y)

        x_out[:, i] = x0[:]
        y_out[:, i] = y0[:]
        u_out[:, i] = u0[:]

        data1[aaim, i, 0:4] = x0[:]
        data1[aaim, i, 4] = y0[:]
        data1[aaim, i, 5] = ysp
        data1[aaim, i, 6] = u0[:]

        x = cp.Variable((n_x, h))
        u = cp.Variable((n_u, h))
        y = cp.Variable((n_y, h))

        cost = 0
        constr = []

        if i == 0:
            xi = 0

        K = D.dot(xi).reshape(4)

        for j in range(h - 1):
            if j == 0:
                constr += [x[:, j] == A @ x0 + B @ u0 + K]
                constr += [y[:, j] == H @ x[:, j]]
                cost += cp.quad_form((y[:, j] - ysp), Q_y0) + cp.quad_form((u[:, j] - u0), Q_u)
            else:
                constr += [x[:, j] == A @ x[:, j - 1] + B @ u[:, j - 1]]
                constr += [y[:, j] == H @ x[:, j]]
                constr += [u[:, j] - u[:, j - 1] == 0]
                cost += cp.quad_form((y[:, j] - ysp), Q_y0) + cp.quad_form((u[:, j] - u[:, j - 1]), Q_u0)
                constr += [y[:, j] <= ub_y, lb_y <= y[:, j]]
            constr += [u[:, j] <= ub_u, lb_u <= u[:, j]]

        constr += [x[:, h - 1] == A @ x[:, h - 2] + B @ u[:, h - 2]]
        constr += [y[:, h - 1] == H @ x[:, h - 1]]
        constr += [u[:, h - 1] - u[:, h - 2] == 0]
        constr += [u[:, h - 1] <= ub_u, lb_u <= u[:, h - 1]]
        constr += [y[:, h - 1] <= ub_y, lb_y <= y[:, h - 1]]
        cost += cp.quad_form((y[:, h - 1] - ysp), Q_y) + cp.quad_form(u[:, h - 1] - u[:, h - 2], Q_u0)

        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve()

        u0 = u[:, 0].value
        u1 = u0.reshape(1, 1)
        #predicted
        x0_hat = x0.reshape(4, 1)
        x1_hat = A.dot(x0_hat) + B.dot(u1)
        #updated
        x0 = A.dot(x0_hat) + B.dot(u1) + D.dot(noise).reshape(4, 1)
        y0 = H.dot(x0)

        xi = y0 - H.dot(x1_hat)

        x0 = x0.reshape(4)
        y0 = y0.reshape(1)

        data2[aaim, i, 0] = u0
    # if aaim % 10 == 0:
    #     y_out = y_out.reshape(NTs)  # 601
    #     u_out = u_out.reshape(NTs)  # 601
    #     plt.plot(range(NTs), u_out, 'r', label="MPC-u")
    #     plt.legend()
    #     plt.show()
    #     plt.plot(range(NTs), y_out, 'r', label="MPC-y")
    #     plt.legend()
    #     plt.show()

np.save('./train_data/inject_input2.npy', data1)
np.save('./train_data/inject_output2.npy', data2)
