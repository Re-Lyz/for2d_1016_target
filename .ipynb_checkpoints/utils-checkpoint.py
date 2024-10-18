import torch
import numpy as np

def get_acc(pred, label):
    batch, _ = pred.shape
    correct, total = 0, 0
    for i in range(batch):
        total += 1
        if label[i] == 0 and pred[i, 0] > pred[i, 1]:
            correct += 1
        elif label[i] == 1 and pred[i, 1] > pred[i, 0]:
            correct += 1
        else:
            continue
    return correct, total

class AvgrageMeter(object):
      
	def __init__(self):
		self.reset()

	def reset(self):
		self.avg = 0
		self.sum = 0
		self.cnt = 0
		self.val = 0

	def update(self, correct, total):
		self.sum += correct
		self.cnt += total
		self.avg = self.sum / self.cnt
