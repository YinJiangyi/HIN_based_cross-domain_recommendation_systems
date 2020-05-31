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
    def __init__(self, ratedim, userdim, itemdim, user_metapaths, item_metapaths, user_cd_metapaths, item_cd_metapaths,
                 trainfile, testfile, steps, delta, beta_e, beta_h, beta_p, beta_w, beta_b, reg_u, reg_v,
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

        self.user_cd_metapathnum = len(user_cd_metapaths)
        self.item_cd_metapathnum = len(item_cd_metapaths)

        ######################### loading original HIN embeddings
        self.X, self.user_metapathdims = self.load_embedding(user_metapaths, 'user', self.unum)  # metapath的embedding
        print('Load user embeddings finished.')

        self.Y, self.item_metapathdims = self.load_embedding(item_metapaths, 'item', self.inum)  # metapath的embedding
        print('Load user embeddings finished.')

        self.X_cd, self.user_cd_metapathdims = self.load_embedding(user_cd_metapaths, 'user',
                                                                   self.unum)  # metapath的embedding
        print('Load user embeddings finished.')

        self.Y_cd, self.item_cd_metapathdims = self.load_embedding(item_cd_metapaths, 'item',
                                                                   self.inum)  # metapath的embedding
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
        self.test_items_index = defaultdict(dict)  # 测试集中取评分大于3的作为喜欢的目标项目

        with open(trainfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip('\n').split('\t')
                R_train.append([self.user_id2index[int(user)], self.item_id2index[int(item)], float(rating)])
                self.rated_items_index[self.user_id2index[int(user)]][self.item_id2index[int(item)]] = float(rating)

        with open(testfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip('\n').split('\t')
                # 过滤test中，train里不存在的user和item
                # if int(user) in self.user_id2index.keys() and int(item) in self.item_id2index.keys():
                if int(user) in self.user_id2index.keys() and int(item) in self.item_id2index.keys() and eval(rating)>3:
                    R_test.append([self.user_id2index[int(user)], self.item_id2index[int(item)], rating])
                    self.test_items_index[self.user_id2index[int(user)]][self.item_id2index[int(item)]] = float(rating)

        print('num of user in test_data is {}'.format(len(self.test_items_index)))

        return R_train, R_test

    def initialize(self):
        self.U = np.random.randn(self.unum, self.ratedim) * 0.1
        self.V = np.random.randn(self.inum, self.ratedim) * 0.1  # 矩阵分解结果

        # self.E = np.random.randn(self.unum, self.itemdim) * 0.1
        # self.H = np.random.randn(self.inum, self.userdim) * 0.1  # 本域metapath_embedding的融合对偶
        #
        # self.E_cd = np.random.randn(self.unum, self.itemdim) * 0.1
        # self.H_cd = np.random.randn(self.inum, self.userdim) * 0.1  # 跨域域metapath_embedding的融合对偶

        self.E = np.random.randn(self.unum, self.itemdim*self.item_metapathnum) * 0.1
        self.H = np.random.randn(self.inum, self.userdim*self.user_metapathnum) * 0.1  # 本域metapath_embedding的融合对偶

        self.E_cd = np.random.randn(self.unum, self.itemdim*self.item_cd_metapathnum) * 0.1
        self.H_cd = np.random.randn(self.inum, self.userdim*self.user_cd_metapathnum) * 0.1  # 跨域域metapath_embedding的融合对偶

        self.pu = np.ones((self.unum, self.user_metapathnum)) * 1.0 / self.user_metapathnum  # 每一行代表user对应的metapath的权值
        self.pv = np.ones((self.inum, self.item_metapathnum)) * 1.0 / self.item_metapathnum  # 每一行代表item对应的metapath的权值

        self.pu_cd = np.ones(
            (self.unum, self.user_cd_metapathnum)) * 1.0 / self.user_cd_metapathnum  # 每一行代表user对应的metapath的权值
        self.pv_cd = np.ones(
            (self.inum, self.item_cd_metapathnum)) * 1.0 / self.item_cd_metapathnum  # 每一行代表item对应的metapath的权值

        self.Mu = {}  # fusion 函数中的转移矩阵，shape：D(latent dim)*d(metapath_dim)
        self.bu = {}

        self.Mu_cd = {}
        self.bu_cd = {}

        for k in range(self.user_metapathnum):
            self.Mu[k] = np.random.randn(self.userdim, self.user_metapathdims[k]) * 0.1
            self.bu[k] = np.random.randn(self.userdim) * 0.1

        for k in range(self.user_cd_metapathnum):
            self.Mu_cd[k] = np.random.randn(self.userdim, self.user_cd_metapathdims[k]) * 0.1
            self.bu_cd[k] = np.random.randn(self.userdim) * 0.1

        self.Mv = {}
        self.bv = {}

        self.Mv_cd = {}
        self.bv_cd = {}

        for k in range(self.item_metapathnum):
            self.Mv[k] = np.random.randn(self.itemdim, self.item_metapathdims[k]) * 0.1
            self.bv[k] = np.random.randn(self.itemdim) * 0.1

        for k in range(self.item_cd_metapathnum):
            self.Mv_cd[k] = np.random.randn(self.itemdim, self.item_cd_metapathdims[k]) * 0.1
            self.bv_cd[k] = np.random.randn(self.itemdim) * 0.1

    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))

    def cal_u(self, i):
        # ui = np.zeros(self.userdim)
        # for k in range(self.user_metapathnum):
        #     ui += self.pu[i][k] * self.sigmod((self.Mu[k].dot(self.X[i][k]) + self.bu[k]))
        # return self.sigmod(ui)
        ui = np.array([])
        for k in range(self.user_metapathnum):
            ui = np.append(ui, self.pu[i][k] * self.sigmod((self.Mu[k].dot(self.X[i][k]) + self.bu[k])))
        return self.sigmod(ui)

    def cal_v(self, j):
        # vj = np.zeros(self.itemdim)
        # for k in range(self.item_metapathnum):
        #     vj += self.pv[j][k] * self.sigmod((self.Mv[k].dot(self.Y[j][k]) + self.bv[k]))
        # return self.sigmod(vj)
        vj = np.array([])
        for k in range(self.item_metapathnum):
            vj = np.append(vj, self.pv[j][k] * self.sigmod((self.Mv[k].dot(self.Y[j][k]) + self.bv[k])))
        return self.sigmod(vj)

    def cal_u_cd(self, i):
        # ui = np.zeros(self.userdim)
        # for k in range(self.user_cd_metapathnum):
        #     ui += self.pu_cd[i][k] * self.sigmod((self.Mu_cd[k].dot(self.X_cd[i][k]) + self.bu_cd[k]))
        # return self.sigmod(ui)
        ui = np.array([])
        for k in range(self.user_cd_metapathnum):
            ui = np.append(ui, self.pu_cd[i][k] * self.sigmod((self.Mu_cd[k].dot(self.X_cd[i][k]) + self.bu_cd[k])))
        return self.sigmod(ui)

    def cal_v_cd(self, j):
        # vj = np.zeros(self.itemdim)
        # for k in range(self.item_cd_metapathnum):
        #     vj += self.pv_cd[j][k] * self.sigmod((self.Mv_cd[k].dot(self.Y_cd[j][k]) + self.bv_cd[k]))
        # return self.sigmod(vj)
        vj = np.array([])
        for k in range(self.item_cd_metapathnum):
            vj = np.append(vj, self.pv_cd[j][k] * self.sigmod((self.Mv_cd[k].dot(self.Y_cd[j][k]) + self.bv_cd[k])))
        return self.sigmod(vj)

    def get_rating(self, i, j):
        ui = self.cal_u(i)
        vj = self.cal_v(j)
        ui_cd = self.cal_u_cd(i)
        vj_cd = self.cal_v_cd(j)
        return self.U[i, :].dot(self.V[j, :]) + self.reg_u * ui.dot(self.H[j, :]) + self.reg_v * self.E[i, :].dot(vj) \
               + self.reg_u * ui_cd.dot(self.H_cd[j, :]) + self.reg_v * self.E_cd[i, :].dot(vj_cd)

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

    def is_early_stop(self, metric='pre'):
        if metric == 'pre':
            record = self.precisons_for_earlystop
        elif metric == 'rec':
            record = self.recalls_for_earlystop
        else:
            print('wrong earlystop metric')
            exit()

        if len(record)==0:
            return False

        if len(record)-record.index(max(record)) > self.early_stop_num_iter:
            return True
        else:
            return False

    def Rec_list_predict(self, sample_user_num=100):
        '''
        对测试集用户进行评分预测
        :param sample_user_num: 用于测试的用户数，默认使用全部测试集用户
        :return: 返回字典格式用户：[推荐列表],[测试集目标列表]
        '''
        # 测试用户抽样
        test_user_list = list(self.test_items_index.keys())
        if len(self.test_items_index.keys())<sample_user_num:
            test_user_list = list(self.test_items_index.keys())
        else:
            test_user_list = random.sample(list(self.test_items_index.keys()), sample_user_num)

        Rec_list={}
        for user in test_user_list:
            u_pre_ratings = {}
            u_unrated_items = set(self.item_list).difference(set(self.rated_items_index[user]))
            for item in u_pre_ratings:
                u_pre_ratings[item] = self.get_rating(user, item)

            u_top_rec_list = nlargest(self.top_K, u_pre_ratings, key=u_pre_ratings.get)
            Rec_list[user]=[[u_top_rec_list],[self.test_items_index[user]]]
        return Rec_list

    def cal_final_PreRec(self, Rec_Tru_list):
        pres,recs=[],[]
        for i in Rec_Tru_list:
            Rec_list, Tru_list = i
            results = self.cal_PR(Rec_list, Tru_list)
            pres.append(results[0])
            recs.append(results[1])
        return float(sum(pres)/len(pres)), float(sum(recs)/len(recs))

    def cal_PR(self, Rec_list, Tru_list):
        hits = [i for i in Rec_list if i in Tru_list]
        Pre = float(len(hits)/len(Rec_list))
        Rec = float(len(hits)/len(Tru_list))
        return Pre, Rec

    def recommend(self):
        self.precisons_for_earlystop, self.recalls_for_earlystop = [],[]   # 用来确定哪个step之后earlystop
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
                ui_cd = self.cal_u_cd(i)
                for k in range(self.user_metapathnum):
                    ui_k = ui[self.userdim*k:self.userdim*(k+1)]
                    x_t = self.sigmod(self.Mu[k].dot(self.X[i][k]) + self.bu[k])
                    pu_g = self.reg_u * -eij * (ui_k * (1 - ui_k) * self.H[j, :][self.userdim*k:self.userdim*(k+1)]).dot(x_t) + self.beta_p * self.pu[i][k]
                    Mu_g = self.reg_u * -eij * self.pu[i][k] * np.array(
                        [ui_k * (1 - ui_k) * x_t * (1 - x_t) * self.H[j, :][self.userdim*k:self.userdim*(k+1)]]).T.dot(
                        np.array([self.X[i][k]])) + self.beta_w * self.Mu[k]
                    bu_g = self.reg_u * -eij * ui_k * (1 - ui_k) * self.pu[i][k] * self.H[j, :][self.userdim*k:self.userdim*(k+1)] * x_t * (
                                1 - x_t) + self.beta_b * self.bu[k]
                    # printpu_g
                    self.pu[i][k] -= 0.1 * self.delta * pu_g
                    self.Mu[k] -= 0.1 * self.delta * Mu_g
                    self.bu[k] -= 0.1 * self.delta * bu_g

                for k in range(self.user_cd_metapathnum):
                    ui_cd_k = ui_cd[self.userdim * k:self.userdim * (k + 1)]
                    x_t_cd = self.sigmod(self.Mu_cd[k].dot(self.X_cd[i][k]) + self.bu_cd[k])
                    pu_g_cd = self.reg_u * -eij * (ui_cd_k * (1 - ui_cd_k) * self.H_cd[j, :][self.userdim * k:self.userdim * (k + 1)]).dot(x_t_cd) + self.beta_p * \
                              self.pu_cd[i][k]
                    Mu_g_cd = self.reg_u * -eij * self.pu_cd[i][k] * np.array(
                        [ui_cd_k * (1 - ui_cd_k) * x_t_cd * (1 - x_t_cd) * self.H_cd[j, :][self.userdim * k:self.userdim * (k + 1)]]).T.dot(
                        np.array([self.X_cd[i][k]])) + self.beta_w * self.Mu_cd[k]
                    bu_g_cd = self.reg_u * -eij * ui_cd_k * (1 - ui_cd_k) * self.pu_cd[i][k] * self.H_cd[j, :][self.userdim * k:self.userdim * (k + 1)] * x_t_cd * (
                            1 - x_t_cd) + self.beta_b * self.bu_cd[k]
                    # printpu_g
                    self.pu_cd[i][k] -= 0.1 * self.delta * pu_g_cd
                    self.Mu_cd[k] -= 0.1 * self.delta * Mu_g_cd
                    self.bu_cd[k] -= 0.1 * self.delta * bu_g_cd

                H_g = self.reg_u * -eij * ui + self.beta_h * self.H[j, :]
                self.H[j, :] -= self.delta * H_g

                H_g_cd = self.reg_u * -eij * ui_cd + self.beta_h * self.H_cd[j, :]
                self.H_cd[j, :] -= self.delta * H_g_cd

                vj = self.cal_v(j)
                vj_cd = self.cal_v_cd(j)
                for k in range(self.item_metapathnum):
                    vj_k = vj[self.itemdim * k:self.itemdim * (k + 1)]
                    y_t = self.sigmod(self.Mv[k].dot(self.Y[j][k]) + self.bv[k])
                    pv_g = self.reg_v * -eij * (vj_k * (1 - vj_k) * self.E[i, :][self.itemdim * k:self.itemdim * (k + 1)]).dot(y_t) + self.beta_p * self.pv[j][k]
                    Mv_g = self.reg_v * -eij * self.pv[j][k] * np.array(
                        [vj_k * (1 - vj_k) * y_t * (1 - y_t) * self.E[i, :][self.itemdim * k:self.itemdim * (k + 1)]]).T.dot(
                        np.array([self.Y[j][k]])) + self.beta_w * self.Mv[k]
                    bv_g = self.reg_v * -eij * vj_k * (1 - vj_k) * self.pv[j][k] * self.E[i, :][self.itemdim * k:self.itemdim * (k + 1)] * y_t * (
                                1 - y_t) + self.beta_b * self.bv[k]

                    self.pv[j][k] -= 0.1 * self.delta * pv_g
                    self.Mv[k] -= 0.1 * self.delta * Mv_g
                    self.bv[k] -= 0.1 * self.delta * bv_g

                for k in range(self.item_cd_metapathnum):
                    vj_cd_k = vj_cd[self.itemdim * k:self.itemdim * (k + 1)]
                    y_t_cd = self.sigmod(self.Mv_cd[k].dot(self.Y_cd[j][k]) + self.bv_cd[k])
                    pv_g_cd = self.reg_v * -eij * (vj_cd_k * (1 - vj_cd_k) * self.E_cd[i, :][self.itemdim * k:self.itemdim * (k + 1)]).dot(y_t_cd) + self.beta_p * \
                              self.pv_cd[j][k]
                    Mv_g_cd = self.reg_v * -eij * self.pv_cd[j][k] * np.array(
                        [vj_cd_k * (1 - vj_cd_k) * y_t_cd * (1 - y_t_cd) * self.E_cd[i, :][self.itemdim * k:self.itemdim * (k + 1)]]).T.dot(
                        np.array([self.Y_cd[j][k]])) + self.beta_w * self.Mv_cd[k]
                    bv_g_cd = self.reg_v * -eij * vj_cd_k * (1 - vj_cd_k) * self.pv_cd[j][k] * self.E_cd[i, :][self.itemdim * k:self.itemdim * (k + 1)] * y_t_cd * (
                            1 - y_t_cd) + self.beta_b * self.bv_cd[k]

                    self.pv_cd[j][k] -= 0.1 * self.delta * pv_g_cd
                    self.Mv_cd[k] -= 0.1 * self.delta * Mv_g_cd
                    self.bv_cd[k] -= 0.1 * self.delta * bv_g_cd

                E_g = self.reg_v * -eij * vj + 0.01 * self.E[i, :]
                self.E[i, :] -= self.delta * E_g

                E_g_cd = self.reg_v * -eij * vj_cd + 0.01 * self.E_cd[i, :]
                self.E_cd[i, :] -= self.delta * E_g_cd

            perror = cerror
            cerror = total_error / n

            self.delta = 0.93 * self.delta

            if (abs(perror - cerror) < 0.0001) or self.is_early_stop(metric='pre'):
                global pre_list, rec_list # 多次重复试验用于求平均值的记录，暂时只进行一次实验
                pre_list.append(Precision)
                rec_list.append(Recall)
                break

            ########################################## 测试
            self.test_users_predict = self.Rec_list_predict()
            Precision, Recall = self.cal_final_PreRec(self.test_users_predict.values())

            self.precisons_for_earlystop.append(Precision)
            self.recalls_for_earlystop.append(Recall)
            # print('step:{}, crror:{}, MAE:{}, RMSE:{} '.format(step, sqrt(cerror), MAE, RMSE, time), time.asctime())
            print('step:{}, crror:{}, Pre:{}, Rec:{} '.format(step, sqrt(cerror), Precision, Recall, time), time.asctime())


if __name__ == "__main__":

    ######################参数设置
    ratedim = 10
    userdim = 30
    itemdim = 10
    train_rate = 0.8

    user_metapaths = ['UbBUb', 'UbTUb']
    item_metapaths = ['BUbB', 'BTB']

    user_cd_metapaths = ['UbTMTUb', 'UbTUmTUb']
    item_cd_metapahts = ['BTMTB', 'BTMuTB']

    # user_metapaths = ['UmMUm', 'UmTUm']
    # item_metapaths = ['MUmM', 'MTM']

    # user_cd_metapaths = ['UmTBTUm', 'UmTUbTUm']
    # item_cd_metapaths = ['MTBTM', 'MTBuTM']

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

    early_stop_num_iter = 30
    top_K_rec = 20

    print('train_rate: ', train_rate)
    print('ratedim: ', ratedim, ' userdim: ', userdim, ' itemdim: ', itemdim)
    print('max_steps: ', steps)
    print('delta: ', delta, 'beta_e: ', beta_e, 'beta_h: ', beta_h, 'beta_p: ', beta_p, 'beta_w: ', beta_w, 'beta_b',
          beta_b, 'reg_u', reg_u, 'reg_v', reg_v)

    # MAE_list = []
    # RMSE_list = []

    pre_list = []
    rec_list = []

    for _ in range(1):  # 多次重复试验取平均值
        HNERec(ratedim, userdim, itemdim, user_metapaths, item_metapaths, user_cd_metapaths, item_cd_metapahts,
               trainfile, testfile, steps, delta, beta_e, beta_h, beta_p, beta_w, beta_b, reg_u, reg_v,
               early_stop_num_iter, top_K_rec)

    # print("MAE:{}, RMSE:{}".format(sum(MAE_list) / len(MAE_list), sum(RMSE_list) / len(RMSE_list)))
    print("Precision:{}, Recall:{}".format(sum(pre_list) / len(pre_list), sum(rec_list) / len(rec_list)))