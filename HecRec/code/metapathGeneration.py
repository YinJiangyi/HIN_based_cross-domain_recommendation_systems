import sys
import numpy as np
import random
from scipy.sparse import csc_matrix,csr_matrix,lil_matrix


# data中的数据用的是id，（不是index，不连续，没有0）
# 路径生成过程中，直接用id作为矩阵索引，不影响，矩阵shape取最大id即可

class metapathGeneration:
    def __init__(self, umnum=5000+1, mnum=5095+1, tnum=699, ubnum=5000+1, bnum=16893+1):
        # 5000+1是指movie user的id从1开始，最大的是5000，所以需要矩阵维度5000+1维，其他类似
        # 699是指tag的编号是从0-698共699维
        self.umnum = umnum
        self.mnum = mnum
        self.tnum = tnum
        self.ubnum = ubnum
        self.bnum = bnum

        # ui = self.load_um('../data/UmM_0.8.train_for_metapath.txt')
        # it = self.load_mt('../data/MT.txt')
        # self.get_MUmM(ui, '../data/metapaths/MUmM.txt')
        # self.get_MTM(it, '../data/metapaths/MTM.txt')
        # self.get_MTBTM(it, '../data/BT.txt', '../data/metapaths/MTBTM.txt')
        # self.get_MTBuTM(it, '../data/UbT.txt', '../data/metapaths/MTBuTM.txt')

        # ut = self.load_ubt('../data/UmT.txt')
        # self.get_UbBUb(ui, '../data/metapaths/UmMUm.txt')
        # self.get_UbTUb(ut, '../data/metapaths/UmTUm.txt')
        # self.get_UbTMTUb(ut, '../data/MT.txt', '../data/metapaths/UmTBTUm.txt')
        # self.get_UbTUmTUb(ut, '../data/UmT.txt', '../data/metapaths/UmTUbTUm.txt')

        ui = self.load_ub('../data/UbB_0.8.train_for_metapath')
        it = self.load_bt('../data/BT.txt')
        self.get_BUbB(ui, '../data/metapaths/BUbB.txt')
        self.get_BTB(it, '../data/metapaths/BTB.txt')
        self.get_BTMTB(it, '../data/MT.txt', '../data/metapaths/BTMTB.txt')
        self.get_BTMuTB(it, '../data/UmT.txt', '../data/metapaths/BTMuTB.txt')

        ut = self.load_ubt('../data/UbT.txt')
        self.get_UbBUb(ui, '../data/metapaths/UbBUb.txt')
        self.get_UbTUb(ut, '../data/metapaths/UbTUb.txt')
        self.get_UbTMTUb(ut, '../data/MT.txt', '../data/metapaths/UbTMTUb.txt')
        self.get_UbTUmTUb(ut, '../data/UmT.txt', '../data/metapaths/UbTUmTUb.txt')


    def load_um(self, ubfile):  # user-item matrix
        ui = np.zeros((self.umnum, self.mnum))
        with open(ubfile, 'r') as infile:
            for line in infile.readlines():
                user, item, rating = line.strip('\n').split(' ')
                ui[int(user)][int(item)] = 1
        return ui

    def load_mt(self, itfile):  # item-tag matrix
        it = np.zeros((self.mnum, self.tnum))
        with open(itfile, 'r') as infile:
            for line in infile.readlines():
                item, tag = line.strip().split('\t')
                it[int(item)][int(tag)] = 1
        return it

    def load_ub(self, ubfile):  # user-item matrix
        ui = np.zeros((self.ubnum, self.bnum))
        with open(ubfile, 'r') as infile:
            for line in infile.readlines():
                user, item, rating = line.strip('\n').split('\t')
                ui[int(user)][int(item)] = 1
        return ui

    def load_bt(self, itfile):  # item-tag matrix
        it = np.zeros((self.bnum, self.tnum))
        with open(itfile, 'r') as infile:
            for line in infile.readlines():
                item, tag = line.strip().split('\t')
                it[int(item)][int(tag)] = 1
        return it

    def load_ubt(self, ubtfile):  # item-tag matrix
        it = np.zeros((self.ubnum, self.tnum))
        with open(ubtfile, 'r') as infile:
            for line in infile.readlines():
                item, tag = line.strip().split('\t')
                it[int(item)][int(tag)] = 1
        return it


    def get_MUmM(self, ui, targetFile):
        print("MUmM...")
        csc_ui = csc_matrix(ui)
        MM = csc_ui.T.dot(csc_ui).toarray()
        lil_MM = lil_matrix(MM)
        print(MM.shape)
        print('writing to file...')
        total = 0

        with open(targetFile, 'w') as outfile:
            for row in range(len(lil_MM.rows)):
                for col in lil_MM.rows[row]:
                    if MM[row][col] != 0 and row != col:
                        outfile.write(str(int(row)) + '\t' + str(int(col)) + '\t' + str(int(MM[row][col])) + '\n')
                        total += 1
        print('total = ', total)

    def get_MTM(self, it, targetFile):
        print("MTM...")
        csc_it = csc_matrix(it)
        MM = csc_it.dot(csc_it.T).toarray()
        lil_MM = lil_matrix(MM)
        print(MM.shape)
        print('writing to file...')
        total = 0

        with open(targetFile, 'w') as outfile:
            for row in range(len(lil_MM.rows)):
                for col in lil_MM.rows[row]:
                    if MM[row][col] != 0 and row != col:
                        outfile.write(str(int(row)) + '\t' + str(int(col)) + '\t' + str(int(MM[row][col])) + '\n')
                        total += 1
        print('total = ', total)


    def get_MTBTM(self, it, btFile, targetFile):
        print('MTBTM...')
        csc_it = csc_matrix(it)
        csr_it = csr_matrix(it)

        bt = np.zeros((self.bnum, self.tnum))
        with open(btFile, 'r') as infile:
            for line in infile.readlines():
                item, tag = line.strip().split('\t')
                bt[int(item)][int(tag)] = 1
        csr_bt = csr_matrix(bt)
        csc_bt = csc_matrix(bt)

        MM = csr_it.dot(csr_bt.T).dot(csr_bt).tocsc().dot(csc_it.T).toarray()
        lil_MM = lil_matrix(MM)
        print(MM.shape)
        print('writing to file...')
        total = 0

        with open(targetFile, 'w') as outfile:
            for row in range(len(lil_MM.rows)):
                for col in lil_MM.rows[row]:
                    if MM[row][col] != 0 and row != col:
                        outfile.write(str(int(row)) + '\t' + str(int(col)) + '\t' + str(int(MM[row][col])) + '\n')
                        total += 1
        print('total = ', total)


    def get_MTBuTM(self, it, ubtFile, targetFile):
        print('MTBuTM...')
        csc_it = csc_matrix(it)

        ubt = np.zeros((self.ubnum, self.tnum))
        with open(ubtFile, 'r') as infile:
            for line in infile.readlines():
                user, tag = line.strip().split('\t')
                ubt[int(user)][int(tag)] = 1
        csc_ubt = csc_matrix(ubt)

        MM = csc_it.dot(csc_ubt.T).dot(csc_ubt).dot(csc_it.T).toarray()
        lil_MM = lil_matrix(MM)
        print(MM.shape)
        print('writing to file...')
        total = 0

        with open(targetFile, 'w') as outfile:
            for row in range(len(lil_MM.rows)):
                for col in lil_MM.rows[row]:
                    if MM[row][col] != 0 and row != col:
                        outfile.write(str(int(row)) + '\t' + str(int(col)) + '\t' + str(int(MM[row][col])) + '\n')
                        total += 1
        print('total = ', total)


    def get_BUbB(self, ui, targetFile):
        print("BUbB...")
        csc_ui = csc_matrix(ui)
        MM = csc_ui.T.dot(csc_ui).toarray()
        lil_MM = lil_matrix(MM)
        print(MM.shape)
        print('writing to file...')
        total = 0

        with open(targetFile, 'w') as outfile:
            for row in range(len(lil_MM.rows)):
                for col in lil_MM.rows[row]:
                    if MM[row][col] != 0 and row != col:
                        outfile.write(str(int(row)) + '\t' + str(int(col)) + '\t' + str(int(MM[row][col])) + '\n')
                        total += 1
        print('total = ', total)

    def get_BTB(self, it, targetFile):
        print("BTB...")
        csc_it = csc_matrix(it)
        MM = csc_it.dot(csc_it.T).toarray()
        lil_MM = lil_matrix(MM)
        print(MM.shape)
        print('writing to file...')
        total = 0

        with open(targetFile, 'w') as outfile:
            for row in range(len(lil_MM.rows)):
                for col in lil_MM.rows[row]:
                    if MM[row][col] != 0 and row != col:
                        outfile.write(str(int(row)) + '\t' + str(int(col)) + '\t' + str(int(MM[row][col])) + '\n')
                        total += 1
        print('total = ', total)


    def get_BTMTB(self, it, mtFile, targetFile):
        print('BTMTB...')
        csc_it = csc_matrix(it)
        csr_it = csr_matrix(it)

        bt = np.zeros((self.mnum, self.tnum))
        with open(mtFile, 'r') as infile:
            for line in infile.readlines():
                item, tag = line.strip().split('\t')
                bt[int(item)][int(tag)] = 1
        csr_bt = csr_matrix(bt)
        csc_bt = csc_matrix(bt)

        MM = csr_it.dot(csr_bt.T).dot(csr_bt).tocsc().dot(csc_it.T).toarray()
        lil_MM = lil_matrix(MM)
        print(MM.shape)
        print('writing to file...')
        total = 0

        with open(targetFile, 'w') as outfile:
            for row in range(len(lil_MM.rows)):
                for col in lil_MM.rows[row]:
                    if MM[row][col] != 0 and row != col:
                        outfile.write(str(int(row)) + '\t' + str(int(col)) + '\t' + str(int(MM[row][col])) + '\n')
                        total += 1
        print('total = ', total)


    def get_BTMuTB(self, it, umtFile, targetFile):
        print('BTMuTB...')
        csc_it = csc_matrix(it)

        ubt = np.zeros((self.umnum, self.tnum))
        with open(umtFile, 'r') as infile:
            for line in infile.readlines():
                user, tag = line.strip().split('\t')
                ubt[int(user)][int(tag)] = 1
        csc_ubt = csc_matrix(ubt)

        MM = csc_it.dot(csc_ubt.T).dot(csc_ubt).dot(csc_it.T).toarray()
        lil_MM = lil_matrix(MM)
        print(MM.shape)
        print('writing to file...')
        total = 0

        with open(targetFile, 'w') as outfile:
            for row in range(len(lil_MM.rows)):
                for col in lil_MM.rows[row]:
                    if MM[row][col] != 0 and row != col:
                        outfile.write(str(int(row)) + '\t' + str(int(col)) + '\t' + str(int(MM[row][col])) + '\n')
                        total += 1
        print('total = ', total)


    def get_UbBUb(self,ui, targetFile):
        print("UbBUb...")
        csc_ui = csc_matrix(ui)
        MM = csc_ui.dot(csc_ui.T).toarray()
        lil_MM = lil_matrix(MM)
        print(MM.shape)
        print('writing to file...')
        total = 0

        with open(targetFile, 'w') as outfile:
            for row in range(len(lil_MM.rows)):
                for col in lil_MM.rows[row]:
                    if MM[row][col] != 0 and row != col:
                        outfile.write(str(int(row)) + '\t' + str(int(col)) + '\t' + str(int(MM[row][col])) + '\n')
                        total += 1
        print('total = ', total)


    def get_UbTUb(self, ut, targetFile):
        print("UbTUb...")
        csc_ut = csc_matrix(ut)
        MM = csc_ut.dot(csc_ut.T).toarray()
        lil_MM = lil_matrix(MM)
        print(MM.shape)
        print('writing to file...')
        total = 0

        with open(targetFile, 'w') as outfile:
            for row in range(len(lil_MM.rows)):
                for col in lil_MM.rows[row]:
                    if MM[row][col] != 0 and row != col:
                        outfile.write(str(int(row)) + '\t' + str(int(col)) + '\t' + str(int(MM[row][col])) + '\n')
                        total += 1
        print('total = ', total)


    def get_UbTMTUb(self, ut, mtFile, targetFile):
        print('UbTMTUb...')
        csc_ut = csc_matrix(ut)

        mt = np.zeros((self.mnum, self.tnum))
        with open(mtFile, 'r') as infile:
            for line in infile.readlines():
                item, tag = line.strip().split('\t')
                mt[int(item)][int(tag)] = 1
        csc_mt = csc_matrix(mt)

        MM = csc_ut.dot(csc_mt.T).dot(csc_mt).dot(csc_ut.T).toarray()
        lil_MM = lil_matrix(MM)
        print(MM.shape)
        print('writing to file...')
        total = 0

        with open(targetFile, 'w') as outfile:
            for row in range(len(lil_MM.rows)):
                for col in lil_MM.rows[row]:
                    if MM[row][col] != 0 and row != col:
                        outfile.write(str(int(row)) + '\t' + str(int(col)) + '\t' + str(int(MM[row][col])) + '\n')
                        total += 1
        print('total = ', total)


    def get_UbTUmTUb(self, ut, umtFile, targetFile):
        print('UbTUmTUb...')
        csc_ut = csc_matrix(ut)

        umt = np.zeros((self.umnum, self.tnum))
        with open(umtFile, 'r') as infile:
            for line in infile.readlines():
                user, tag = line.strip().split('\t')
                umt[int(user)][int(tag)] = 1
        csc_umt = csc_matrix(umt)

        MM = csc_ut.dot(csc_umt.T).dot(csc_umt).dot(csc_ut.T).toarray()
        lil_MM = lil_matrix(MM)
        print(MM.shape)
        print('writing to file...')
        total = 0

        with open(targetFile, 'w') as outfile:
            for row in range(len(lil_MM.rows)):
                for col in lil_MM.rows[row]:
                    if MM[row][col] != 0 and row != col:
                        outfile.write(str(int(row)) + '\t' + str(int(col)) + '\t' + str(int(MM[row][col])) + '\n')
                        total += 1
        print('total = ', total)


if __name__ == '__main__':
    metapathGeneration()
