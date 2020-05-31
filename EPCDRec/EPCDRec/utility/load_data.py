'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import asctime, time

class Data(object):
    def __init__(self, path, batch_size, is_HIN=0):
        self.path = path
        self.batch_size = batch_size
        self.is_HIN = is_HIN

        train_file = path + '/train.txt'   # 数据每行第一个是用户id，后面是item_id
                                           # test中的user全部在train中存在过，item可能不

        u_t = path + '/U_tag.txt'
        i_t = path + '/I_tag.txt'

        ia_t = path + '/Ib_tag.txt'  # auxiliary IT & UT
        ua_t = path + '/Ub_tag.txt'


        test_file = path + '/test.txt'

        #get number of users and items
        self.n_users, self.n_items, self.n_tags = 0, 0, 0
        self.n_train, self.n_test = 0, 0  # 所有的u-i对个数
        self.neg_pools = {}

        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    try:
                        items = [int(i) for i in l[1:]]
                    except:
                        print(l)
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))  # 只更新item个数
                    self.n_test += len(items)

        with open(u_t) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    tags = [int(i) for i in l[1:]]
                    self.n_tags = max(self.n_tags, max(tags))
        with open(i_t) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    tags = [int(i) for i in l[1:]]
                    self.n_tags = max(self.n_tags, max(tags))

        self.n_items += 1  # index从0开始
        self.n_users += 1
        self.n_tags += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)  # R里包含train和test中所有的user和item
        self.U_T = sp.dok_matrix((self.n_users, self.n_tags), dtype=np.float32)
        self.I_T = sp.dok_matrix((self.n_items, self.n_tags), dtype=np.float32)

        self.train_set, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_set = items[0], items[1:]   # train数据集中，user后可以没有任何item

                    for i in train_set:
                        self.R[uid, i] = 1.
                        # self.R[uid][i] = 1

                    self.train_set[uid] = train_set

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        with open(u_t) as f_ut:
            for l in f_ut.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                tags = [int(i) for i in l.split(' ')]
                uid, tag_set = tags[0], tags[1:]   # train数据集中，user后可以没有任何item

                for i in tag_set:
                    self.U_T[uid, i] = 1.


        with open(i_t) as f_it:
            for l in f_it.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                tags = [int(i) for i in l.split(' ')]
                iid, tag_set = tags[0], tags[1:]   # train数据集中，user后可以没有任何item

                for i in tag_set:
                    self.I_T[iid, i] = 1.


        # cross-domain mats
        # 辅助域相关的user和item个数（计算mat的维度）
        ia_set, ua_set = [], []
        with open(ia_t) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    ia_set.append(int(l[0]))
        with open(ua_t) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    ua_set.append(int(l[0]))

        self.n_ia = max(ia_set) + 1
        self.n_ua = max(ua_set) + 1

        self.Ua_T = sp.dok_matrix((self.n_ua, self.n_tags), dtype=np.float32)
        self.Ia_T = sp.dok_matrix((self.n_ia, self.n_tags), dtype=np.float32)

        with open(ua_t) as f:
            for l in f.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                tags = [int(i) for i in l.split(' ')]
                uid, tag_set = tags[0], tags[1:]

                for i in tag_set:
                    self.Ua_T[uid, i] = 1.

        with open(ia_t) as f:
            for l in f.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                tags = [int(i) for i in l.split(' ')]
                iid, tag_set = tags[0], tags[1:]

                for i in tag_set:
                    self.Ia_T[iid, i] = 1.



    def get_adj_mat(self):
        try:
            print(asctime())
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')

            uti_adj_mat = sp.load_npz(self.path + '/s_uti_adj_mat.npz')
            uti_norm_adj_mat = sp.load_npz(self.path + '/s_uti_norm_adj_mat.npz')
            uti_mean_adj_mat = sp.load_npz(self.path + '/s_uti_mean_adj_mat.npz')

            cross_domain_mats = {}
            cross_domain_mats['utiati_adj_mat'], cross_domain_mats['utiati_norm_adj_mat'], cross_domain_mats[
                'utiati_mean_adj_mat'] = sp.load_npz(self.path + '/s_utiati_adj_mat.npz'), \
                                         sp.load_npz(self.path + '/s_utiati_norm_adj_mat.npz'), \
                                         sp.load_npz(self.path + '/s_utiati_mean_adj_mat.npz')

            cross_domain_mats['utuati_adj_mat'], cross_domain_mats['utuati_norm_adj_mat'], cross_domain_mats[
                'utuati_mean_adj_mat'] = sp.load_npz(self.path + '/s_utuati_adj_mat.npz'), \
                                         sp.load_npz(self.path + '/s_utuati_norm_adj_mat.npz'), \
                                         sp.load_npz(self.path + '/s_utuati_mean_adj_mat.npz')


            print('already load adj matrix and meta-path_mat', adj_mat.shape, time() - t1)
            print(asctime())

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat(self.R)

            # 本域uti
            uti_mat = self.U_T.dot(self.I_T.T)
            # 过滤原有的UI对
            for uid in self.test_set.keys():
                for iid in self.test_set[uid]:
                    uti_mat[(uid, iid)]=0
            # uti_mat转化为01矩阵
            uti_mat[uti_mat.nonzero()] = 1

            uti_adj_mat, uti_norm_adj_mat, uti_mean_adj_mat = self.create_adj_mat(uti_mat)

            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)

            sp.save_npz(self.path + '/s_uti_adj_mat.npz', uti_adj_mat)
            sp.save_npz(self.path + '/s_uti_norm_adj_mat.npz', uti_norm_adj_mat)
            sp.save_npz(self.path + '/s_uti_mean_adj_mat.npz', uti_mean_adj_mat)

            # 跨域meta-path
            cross_domain_mats = {}

            utiati_mat = self.U_T.dot(self.Ia_T.T).dot(self.Ia_T).dot(self.I_T.T)
            # 过滤原有的UI对
            for uid in self.test_set.keys():
                for iid in self.test_set[uid]:
                    utiati_mat[(uid, iid)] = 0
            # uti_mat转化为01矩阵
            utiati_mat[utiati_mat.nonzero()] = 1
            cross_domain_mats['utiati_adj_mat'], cross_domain_mats['utiati_norm_adj_mat'], cross_domain_mats[
                'utiati_mean_adj_mat'] = self.create_adj_mat(utiati_mat)

            utuati_mat = self.U_T.dot(self.Ua_T.T).dot(self.Ua_T).dot(self.I_T.T)
            # 过滤原有的UI对
            for uid in self.test_set.keys():
                for iid in self.test_set[uid]:
                    utuati_mat[(uid, iid)] = 0
            # uti_mat转化为01矩阵
            utuati_mat[utuati_mat.nonzero()] = 1
            cross_domain_mats['utuati_adj_mat'], cross_domain_mats['utuati_norm_adj_mat'], cross_domain_mats[
                'utuati_mean_adj_mat'] = self.create_adj_mat(utuati_mat)

            sp.save_npz(self.path + '/s_utiati_adj_mat.npz', cross_domain_mats['utiati_adj_mat'])
            sp.save_npz(self.path + '/s_utiati_norm_adj_mat.npz', cross_domain_mats['utiati_norm_adj_mat'])
            sp.save_npz(self.path + '/s_utiati_mean_adj_mat.npz', cross_domain_mats['utiati_mean_adj_mat'])

            sp.save_npz(self.path + '/s_utuati_adj_mat.npz', cross_domain_mats['utuati_adj_mat'])
            sp.save_npz(self.path + '/s_utuati_norm_adj_mat.npz', cross_domain_mats['utuati_norm_adj_mat'])
            sp.save_npz(self.path + '/s_utuati_mean_adj_mat.npz', cross_domain_mats['utuati_mean_adj_mat'])


        return adj_mat, norm_adj_mat, mean_adj_mat, uti_adj_mat, uti_norm_adj_mat, uti_mean_adj_mat, cross_domain_mats



    def create_adj_mat(self, adj_matrix):  # 实现graph Laplacian norm pui
        t1 = time()

        n_row, n_col = adj_matrix.shape
        adj_mat = sp.dok_matrix((n_row + n_col, n_row + n_col), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = adj_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))  # 邻居个数：user交互过的所有item的个数+1 和 item被交互过的user的个数+1

            d_inv = np.power(rowsum, -1).flatten()  # 求倒数
            d_inv[np.isinf(d_inv)] = 0.  # 如果存在求倒数之后有无穷大，则置0
            d_mat_inv = sp.diags(d_inv)  # 以d_inv中的元素构造对角元素

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()


    def negative_pool(self):
        t1 = time()
        for u in self.train_set.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_set[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]


        def sample_pos_items_for_u(u, num):
            pos_items = self.train_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_set[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_set[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d, n_tags=%d' % (self.n_users, self.n_items, self.n_tags))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_set[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state
