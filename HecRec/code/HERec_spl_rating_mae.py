#!/usr/bin/python
# encoding=utf-8
import numpy as np
import time
import random
from math import sqrt, fabs, log, exp
import sys
import pandas as pd
from collections import defaultdict
from heapq import nlargest


class HNERec:
    def __init__(self, ratedim, userdim, itemdim, user_metapaths, item_metapaths, trainfile, testfile, steps, delta,
                 beta_e, beta_h, beta_p, beta_w, beta_b, reg_u, reg_v,
                 early_stop_num_iter, top_K_rec):
        self.data_process(trainfile)

        self.ratedim = ratedim
        self.userdim = userdim
        self.itemdim = itemdim
        self.steps = steps
        self.delta = delta
        self.beta_e = beta_e
        self.beta_h = beta_h
        self.beta_p = beta_p
        self.beta_w = beta_w
        self.beta_b = beta_b
        self.reg_u = reg_u
        self.reg_v = reg_v

        self.early_stop_num_iter = early_stop_num_iter
        self.top_K = top_K_rec

        self.user_metapathnum = len(user_metapaths)
        self.item_metapathnum = len(item_metapaths)

        #########################OK
        self.X, self.user_metapathdims = self.load_embedding(user_metapaths, 'user', self.unum)  # metapath的embedding
        print('Load user embeddings finished.')

        self.Y, self.item_metapathdims = self.load_embedding(item_metapaths, 'item', self.inum)  # metapath的embedding
        print('Load user embeddings finished.')
        #########################OK

        self.R, self.T = self.load_rating(trainfile, testfile)
        #########################OK
        print('Load rating finished.')
        print('train size : ', len(self.R))
        print('test size : ', len(self.T))

        self.initialize()
        self.recommend()

    def data_process(self, train_File):
        # 把不连续的user item id替换成连续的index
        train_data = pd.read_csv(train_File, sep='\t', header=None, names=['user_id', 'item_id', 'rating'])
        self.user_list = train_data.user_id.unique()
        self.user_id2index = {j: i for i, j in list(enumerate(self.user_list))}
        self.unum = len(self.user_list)

        self.item_list = train_data.item_id.unique()
        self.item_id2index = {j: i for i, j in list(enumerate(self.item_list))}
        self.inum = len(self.item_list)

        # test_data 里不存在与train里的数据的user和item的过滤放在load_rating里

    def load_embedding(self, metapaths, userORitem, num):
        if userORitem == 'user':
            id2index = self.user_id2index
        else:
            id2index = self.item_id2index

        X = {}
        for i in range(num):
            X[i] = {}
        metapathdims = []

        ctn = 0
        for metapath in metapaths:
            sourcefile = '../data/embeddings/' + metapath
            # printsourcefile
            with open(sourcefile) as infile:

                k = int(infile.readline().strip().split(' ')[1])
                metapathdims.append(k)
                for i in range(num):
                    X[i][ctn] = np.zeros(k)

                n = 0
                for line in infile.readlines():
                    n += 1
                    arr = line.strip().split(' ')
                    ID = int(arr[0])
                    if ID in list(id2index.keys()):  # 可能划分数据集后，train里不存在该id
                        index = id2index[ID]
                        for j in range(k):
                            X[index][ctn][j] = float(arr[j + 1])
                    else:
                        pass
                print('metapath ', metapath, 'numbers ', n)
            ctn += 1
        return X, metapathdims

    def load_rating(self, trainfile, testfile):
        R_train = []
        R_test = []

        self.rated_items_index = defaultdict(dict)
        self.test_items_index = defaultdict(dict)

        with open(trainfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip('\n').split('\t')
                R_train.append([self.user_id2index[int(user)], self.item_id2index[int(item)], float(rating)])
                self.rated_items_index[self.user_id2index[int(user)]][self.item_id2index[int(item)]] = float(rating)

        with open(testfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip('\n').split('\t')
                # 过滤test中，train里不存在的user和item
                if int(user) in self.user_id2index.keys() and int(item) in self.item_id2index.keys():
                    R_test.append([self.user_id2index[int(user)], self.item_id2index[int(item)], rating])
                    self.test_items_index[self.user_id2index[int(user)]][self.item_id2index[int(item)]] = float(rating)

        print('num of user in test_data is {}'.format(len(self.test_items_index)))

        return R_train, R_test

    def initialize(self):
        self.E = np.random.randn(self.unum, self.itemdim) * 0.1
        self.H = np.random.randn(self.inum, self.userdim) * 0.1
        self.U = np.random.randn(self.unum, self.ratedim) * 0.1
        self.V = np.random.randn(self.inum, self.ratedim) * 0.1

        self.pu = np.ones((self.unum, self.user_metapathnum)) * 1.0 / self.user_metapathnum  # 每一行代表user对应的4种metapath的权值
        self.pv = np.ones((self.inum, self.item_metapathnum)) * 1.0 / self.item_metapathnum  # 每一行代表item对应的4种metapath的权值

        self.Mu = {}  # fusion 函数中的转移矩阵，shape：D(latent dim)*d(metapath_dim)
        self.bu = {}
        for k in range(self.user_metapathnum):
            self.Mu[k] = np.random.randn(self.userdim, self.user_metapathdims[k]) * 0.1
            self.bu[k] = np.random.randn(self.userdim) * 0.1

        self.Mv = {}
        self.bv = {}
        for k in range(self.item_metapathnum):
            self.Mv[k] = np.random.randn(self.itemdim, self.item_metapathdims[k]) * 0.1
            self.bv[k] = np.random.randn(self.itemdim) * 0.1

    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))

    def cal_u(self, i):
        ui = np.zeros(self.userdim)
        for k in range(self.user_metapathnum):
            ui += self.pu[i][k] * self.sigmod((self.Mu[k].dot(self.X[i][k]) + self.bu[k]))
        return self.sigmod(ui)

    def cal_v(self, j):
        vj = np.zeros(self.itemdim)
        for k in range(self.item_metapathnum):
            vj += self.pv[j][k] * self.sigmod((self.Mv[k].dot(self.Y[j][k]) + self.bv[k]))
        return self.sigmod(vj)

    def get_rating(self, i, j):
        ui = self.cal_u(i)
        vj = self.cal_v(j)
        return self.U[i, :].dot(self.V[j, :]) + self.reg_u * ui.dot(self.H[j, :]) + self.reg_v * self.E[i, :].dot(vj)

    def maermse(self):
        m = 0.0
        mae = 0.0
        rmse = 0.0
        n = 0
        for t in self.T:
            n += 1
            i = t[0]
            j = t[1]
            r = t[2]
            r_p = self.get_rating(i, j)

            if r_p > 5: r_p = 5
            if r_p < 1: r_p = 1
            m = fabs(float(r_p) - float(r))
            mae += m
            rmse += m * m
        mae = mae * 1.0 / n
        rmse = sqrt(rmse * 1.0 / n)
        return mae, rmse

    def recommend(self, early_stop_metric='mae'):
        mae = []
        rmse = []
        starttime = time.asctime()
        perror = 99999
        cerror = 9999
        n = len(self.R)

        print('start training', time.asctime())
        for step in range(steps):
            total_error = 0.0
            for t in self.R:
                i = t[0]
                j = t[1]
                rij = t[2]

                rij_t = self.get_rating(i, j)
                eij = rij - rij_t
                total_error += eij * eij

                # 更新X，Y矩阵
                U_g = -eij * self.V[j, :] + self.beta_e * self.U[i, :]
                V_g = -eij * self.U[i, :] + self.beta_h * self.V[j, :]

                self.U[i, :] -= self.delta * U_g
                self.V[j, :] -= self.delta * V_g

                ui = self.cal_u(i)  # 计算u的metapath融合后向量
                for k in range(self.user_metapathnum):
                    x_t = self.sigmod(self.Mu[k].dot(self.X[i][k]) + self.bu[k])

                    pu_g = self.reg_u * -eij * (ui * (1 - ui) * self.H[j, :]).dot(x_t) + self.beta_p * self.pu[i][k]

                    Mu_g = self.reg_u * -eij * self.pu[i][k] * np.array(
                        [ui * (1 - ui) * x_t * (1 - x_t) * self.H[j, :]]).T.dot(
                        np.array([self.X[i][k]])) + self.beta_w * self.Mu[k]
                    bu_g = self.reg_u * -eij * ui * (1 - ui) * self.pu[i][k] * self.H[j, :] * x_t * (
                                1 - x_t) + self.beta_b * self.bu[k]
                    # printpu_g
                    self.pu[i][k] -= 0.1 * self.delta * pu_g
                    self.Mu[k] -= 0.1 * self.delta * Mu_g
                    self.bu[k] -= 0.1 * self.delta * bu_g

                H_g = self.reg_u * -eij * ui + self.beta_h * self.H[j, :]
                self.H[j, :] -= self.delta * H_g

                vj = self.cal_v(j)
                for k in range(self.item_metapathnum):
                    y_t = self.sigmod(self.Mv[k].dot(self.Y[j][k]) + self.bv[k])
                    pv_g = self.reg_v * -eij * (vj * (1 - vj) * self.E[i, :]).dot(y_t) + self.beta_p * self.pv[j][k]
                    Mv_g = self.reg_v * -eij * self.pv[j][k] * np.array(
                        [vj * (1 - vj) * y_t * (1 - y_t) * self.E[i, :]]).T.dot(
                        np.array([self.Y[j][k]])) + self.beta_w * self.Mv[k]
                    bv_g = self.reg_v * -eij * vj * (1 - vj) * self.pv[j][k] * self.E[i, :] * y_t * (
                                1 - y_t) + self.beta_b * self.bv[k]

                    self.pv[j][k] -= 0.1 * self.delta * pv_g
                    self.Mv[k] -= 0.1 * self.delta * Mv_g
                    self.bv[k] -= 0.1 * self.delta * bv_g

                E_g = self.reg_v * -eij * vj + 0.01 * self.E[i, :]
                self.E[i, :] -= self.delta * E_g

            perror = cerror
            cerror = total_error / n

            # self.delta = 0.93 * self.delta

            if len(mae) > 2 and (mae[-1] > mae[-2]):
                global MAE_list, RMSE_list
                MAE_list.append(mae[-2])
                RMSE_list.append(rmse[-2])
                break
            # print('step ', step, 'crror : ', sqrt(cerror))
            MAE, RMSE = self.maermse()
            mae.append(MAE)
            rmse.append(RMSE)
            print('step:{}, crror:{}, MAE:{}, RMSE:{} '.format(step, sqrt(cerror), MAE, RMSE, time), time.asctime())
            # endtime = time.asctime()
            # print('time: ', endtime)
        # print('MAE: ', min(mae), ' RMSE: ', min(rmse))


if __name__ == "__main__":
    # unum = 16239
    # inum = 14284

    ratedim = 10
    userdim = 30
    itemdim = 10
    train_rate = 0.8

    # user_metapaths = ['UbBUb', 'UbTUb']
    # item_metapaths = ['BUbB', 'BTB']

    user_metapaths = ['UbBUb', 'UbTUb', 'UbTMTUb', 'UbTUmTUb']
    item_metapaths = ['BUbB', 'BTB', 'BTMTB', 'BTMuTB']
    # user_metapaths = ['UmMUm', 'UmTUm', 'UmTBTUm', 'UmTUbTUm']
    # item_metapaths = ['MUmM', 'MTM', 'MTBTM', 'MTBuTM']
    # user_metapaths = ['UmMUm', 'UmTUm']
    # item_metapaths = ['MUmM', 'MTM']

    trainfile = '../data/UbB_' + str(train_rate) + '.train'
    testfile = '../data/UbB_' + str(train_rate) + '.test'
    steps = 10000
    delta = 0.02  # 初始学习率
    beta_e = 0.1
    beta_h = 0.1
    beta_p = 2
    beta_w = 0.1
    beta_b = 0.1
    reg_u = 1.0
    reg_v = 1.0

    early_stop_num_iter = 10
    top_K_rec = 10

    print('train_rate: ', train_rate)
    print('ratedim: ', ratedim, ' userdim: ', userdim, ' itemdim: ', itemdim)
    print('max_steps: ', steps)
    print('delta: ', delta, 'beta_e: ', beta_e, 'beta_h: ', beta_h, 'beta_p: ', beta_p, 'beta_w: ', beta_w, 'beta_b',
          beta_b, 'reg_u', reg_u, 'reg_v', reg_v)

    MAE_list, RMSE_list = [], []
    for _ in range(5):
        HNERec(ratedim, userdim, itemdim, user_metapaths, item_metapaths, trainfile, testfile, steps, delta, beta_e,
               beta_h, beta_p, beta_w, beta_b, reg_u, reg_v, early_stop_num_iter, top_K_rec)

    print('#############################################################')
    # print(MAE_list, RMSE_list)
    print(sum(MAE_list) / len(MAE_list), sum(RMSE_list) / len(RMSE_list))