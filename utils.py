import csv
import os
import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import numpy as np
class AverageMeter(object):
    '''computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count =0

    def update(self,val,n=1):
        self.val = val
        self.sum += val*n
        self.count +=n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t') #\t为一个tab

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, labels):
    batch_size = labels.size(0)
    _, pred = outputs.topk(k=1, dim=1, largest=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1))
    n_correct_elems = correct.float().sum().data

    return n_correct_elems / batch_size

def calculate_recall(outputs, labels, opt):
    _, pred = outputs.topk(k=1, dim=1, largest=True)
    pred = pred.t() # 转置成行
    if opt.n_classes == 3:
        TP = (((pred.data == 1) & (labels.data == 1)) | ((pred.data == 2) & (labels.data == 2))).cpu().float().sum().data
        FP = (((pred.data == 1) & (labels.data == 0)) | ((pred.data == 2) & (labels.data == 0))).cpu().float().sum().data
        TN = ((pred.data == 0) & (labels.data == 0)).cpu().float().sum().data
        FN = (((pred.data == 0) & (labels.data == 1)) | ((pred.data == 0) & (labels.data == 2))).cpu().float().sum().data

        if sum(labels.data) == 0:
            TP = 1
            FN = 0
        label_0=0
        for label in labels:
            if label == 0:
                label_0 = label_0+1
            else:
                continue
        if label_0 == 0:
            TN = 1
            FP = 0
        if TP + FN == 0:
            r = torch.tensor(0).float()
        else:
            r = TP / (TP + FN)  # recall
        if TP + FP == 0:
            p = torch.tensor(0).float()
        else:
            p = TP / (TP + FP)
        if r + p == 0:
            f1 = torch.tensor(0).float()
        else:
            f1 = 2 * r * p / (r + p)
        if TP + FN == 0:
            sen = torch.tensor(0).float()
        else:
            sen = TP / (TP + FN)
        if TN + FP == 0:
            sp = torch.tensor(0).float()
        else:
            sp = TN / (TN + FP)
        return r, p, f1, sen, sp
    else:
        TP = ((pred.data == 1) & (labels.data == 1)).cpu().float().sum().data
        FP = ((pred.data == 1) & (labels.data == 0)).cpu().float().sum().data
        TN = ((pred.data == 0) & (labels.data == 0)).cpu().float().sum().data
        #TN = ((pred.data == 0) & (labels.data == 0)).cpu().float().sum().data
        FN = (((pred.data == 0) & (labels.data == 1))).cpu().float().sum().data
        if sum(labels.data) == 0:
            TP = 1
            FN = 0
        label_0 = 0
        for label in labels:
            if label == 0:
                label_0 = label_0+1
            else:
                continue
        if label_0 == 0:
            TN = 1
            FP = 0
        if TP + FN == 0:
            r = torch.tensor(0).float()
        else:
            r = TP / (TP + FN)  # recall
            # r = torch.tensor(r).float()
        if TP + FP == 0:
            p = torch.tensor(0).float()
        else:
            p = TP / (TP + FP)
            # p = torch.tensor(p).float()
        if r + p == 0:
            f1 = torch.tensor(0).float()
        else:
            f1 = 2 * r * p / (r + p)
            # f1 = torch.tensor(f1).float()
        if TP + FN == 0:
            sen = torch.tensor(0).float()
        else:
            sen = TP / (TP + FN)
            # sen = torch.tensor(sen).float()
        if TN + FP == 0:
            sp = torch.tensor(0).float()
        else:
            sp = TN / (TN + FP)
            # sp = torch.tensor(sp).float()
        return r, p, f1, sen, sp
def OsJoin(*args):
    p = os.path.join(*args)
    p = p.replace('\\', '/')
    return p
