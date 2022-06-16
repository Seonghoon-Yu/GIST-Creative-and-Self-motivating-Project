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

from get_user_id import mapping_problems

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	else:
		return 0


def ndcg(gt_item, pred_items):
		if gt_item in pred_items:
				index = pred_items.index(gt_item)
				return np.reciprocal(np.log2(index+2))
		return 0

def metrics(model, test_loader, top_k):
		HR, NDCG = [], []

		for user, item, label in test_loader:
				user = user.cuda()
				item = item.cuda()

				predictions = model(user, item)
				_, indices = torch.topk(predictions, top_k)
				recommends = torch.take(item, indices).cpu().numpy().tolist()

				gt_item = item[0].item()
				HR.append(hit(gt_item, recommends))
				NDCG.append(ndcg(gt_item, recommends))

		return np.mean(HR), np.mean(NDCG)


def get_recommends(user, item, model, top_k=1000):
	prediction = model(user, item)
	_, indices = torch.topk(prediction, top_k)
	recommends = torch.take(item, indices).cpu().numpy().tolist()

	return recommends


def post_process_recommends(recommends, mapped_problems, top_k = 5):
	recommends = list(recommends)
	refine_recommends = recommends[:]
	mapped_problems = list(mapped_problems)

	for recommend in recommends:
		if recommend in mapped_problems:
			refine_recommends.remove(recommend)

	recommends = mapping_problems(refine_recommends, reverse=True)[:top_k]

	return recommends


