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

GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'

args = easydict.EasyDict({
    'lr': 0.001,
    'dropout': 0.0,
    'batch_size': 64,
    'epochs': 30,
    'top_k': 10,
    'factor_num': 8,
    'num_layers': 3,
    'num_ng': 8,
    'test_num_ng': 99,
    'out': True
})

cudnn.benchmark = True

#### Preapare Dataset ####
print('Prepare Dataset')
train_data, test_data, user_num, item_num, train_mat = load_all(train_rating, test_negative)

train_dataset = NCFData(train_data, item_num, train_mat, args.num_ng, True)
test_dataset = NCFData(test_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=64)
test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=64)
print('End dataset')

#### Create Model ####
if n_model == 'NeuMF-pre':
    assert os.path.exists(GMF_model_path), 'lack of GMF model'
    assert os.path.exists(MLP_model_path), 'lack of MLP model'
    GMF_model = torch.load(GMF_model_path)
    MLP_model = torch.load(MLP_model_path)
else:
    GMF_model = None
    MLP_model = None

model = NCF(user_num, item_num, args.factor_num, args.num_layers,
            args.dropout, n_model, GMF_model, MLP_model)

model.cuda()
loss_function = nn.BCEWithLogitsLoss()

if n_model == 'NeuMF-pre':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

#### Training ####
print('Start Training')
count, best_hr = 0, 0
for epoch in range(args.epochs):
    model.train()  # Enable dropout (if have).
    start_time = time.time()
    train_loader.dataset.ng_sample()

    for user, item, label in train_loader:
        user = user.cuda()
        item = item.cuda()
        label = label.float().cuda()

        model.zero_grad()
        prediction = model(user, item)
        loss = loss_function(prediction, label)
        loss.backward()
        optimizer.step()
        # writer.add_scalar('data/loss', loss.item(), count)
        count += 1

    model.eval()
    HR, NDCG = metrics(model, test_loader, args.top_k)

    elapsed_time = time.time() - start_time
    print("The time elapse of epoch {:03d}".format(epoch + 1) + " is: " + time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

    if HR > best_hr:
        best_hr, best_ndcg, best_epoch = HR, NDCG, epoch + 1
        if args.out:
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            # torch.save(model.state_dict(), '{}{}.pth'.format(model_path, 'NeuMF'))
            torch.save(model, '{}{}.pth'.format(model_path, 'NeuMF'))

print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))

