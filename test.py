import os
import time
import easydict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from model import NCF
from dataset import load_all, NCFData
from utils import hit, ndcg, metrics

### Config
n_model = 'NeuMF-end'

train_rating = './data/train_rating.csv'
test_rating = './data/test_rating.csv'
test_negative = './data/test_negative.csv'

model_path = './checkpoint/'


NeuMF_model_path = model_path + 'NeuMF.pth'

args = easydict.EasyDict({
    'lr': 0.001,
    'dropout': 0.0,
    'batch_size': 64,
    'epochs': 30,
    'top_k': 1,
    'factor_num': 8,
    'num_layers': 3,
    'num_ng': 8,
    'test_num_ng': 99,
    'out': True
})

cudnn.benchmark = True

#### Preapare Dataset ####
# train_data, test_data, user_num, item_num, train_mat = load_all(train_rating, test_negative)
#
# train_dataset = NCFData(train_data, item_num, train_mat, args.num_ng, True)
# test_dataset = NCFData(test_data, item_num, train_mat, 0, False)
# train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=64)
# test_loader = data.DataLoader(test_dataset, batch_size=50, shuffle=False, num_workers=64)

########################### CREATE MODEL #################################
GMF_model = None
MLP_model = None

my_model = torch.load(NeuMF_model_path)

my_model.cuda()
my_model.eval()

recommend_list = []

# for user, item, label in test_loader:
    # user shape=[batch(100)], value=[1,1,1,1,...,1,1,1]
    # item shape=[batch(100)], value=[1234, 4325, 3523]
    # label shape=[batch(100)], value=[0,0,0,~]

user = torch.tensor([10000,10000])
item = torch.tensor([1044, 1024])

temp = []
user = user.cuda()
item = item.cuda()

predictions = my_model(user, item) # prediction shape = [100]
print(predictions, 'predictions')
_, indices = torch.topk(predictions, args.top_k)
print(indices, 'indices')
recommends = torch.take(item, indices).cpu().numpy().tolist()
print(recommends, 'recommends')

temp.append(user[0].item())
temp.extend(recommends)
recommend_list.append(temp)

# recommend_df = pd.DataFrame(recommend_list)
# recommend_df.to_csv("./result/recommend.csv", header = None, index = None)