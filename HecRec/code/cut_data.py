#!/user/bin/python
import random

train_rate = 0.8

R = []
with open('../data/UbB_all.txt', 'r') as infile:
    for line in infile.readlines():
        user, item, rating = line.strip('\n').split(' ')
        R.append([user, item, rating])

random.shuffle(R)
train_num = int(len(R) * train_rate)

with open('../data/UbB_' + str(train_rate) + '.train', 'w') as trainfile,\
     open('../data/UbB_' + str(train_rate) + '.test', 'w') as testfile, \
     open('../data/UbB_' + str(train_rate) + '.train_for_metapath', 'w') as metapathfile:
     for r in R[:train_num]:
         trainfile.write('\t'.join(r) + '\n')
         if float(r[-1])>3:
             metapathfile.write('\t'.join(r) + '\n')

     for r in R[train_num:]:
         testfile.write('\t'.join(r) + '\n')


