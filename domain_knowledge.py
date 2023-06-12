import numpy as np
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

import argparse
import time
import pickle
import pandas as pd
from tqdm import tqdm
from collections import namedtuple, deque

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support

from dataloader import IEMOCAPDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=30, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True,
                        help='class weight')
    parser.add_argument('--active-listener', action='store_true', default=False,
                        help='active listener')
    parser.add_argument('--attention', default='general', help='Attention type')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')
    parser.add_argument('--emotion_pair_size',default = 3, type=int, help = 'Size of emotion\
    pair for Domain Knowledge')
    parser.add_argument('--path',type=str, default='', help='path to IEMOCAP_features.pkl')
    parser.add_argument('--attribute', type=int, default=1, help='AVEC attribute')
    args = parser.parse_args()

    emotion_pair_size = args.emotion_pair_size

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    n_classes  = 6
    cuda       = args.cuda
    n_epochs   = args.epochs
    numworkers = 0
    n_actions = ['0', '1', '2', '3', '4', '5']
    batch = 10

    sum_iemocap = 0

    
    
    dir_path = "%s%d" % ('AEPR', -1)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    log_file = "%s/print.log" % dir_path
    f = open(log_file, "w+")
    sys.stdout = f

    trainset_iemocap = IEMOCAPDataset(path=args.path) # with full daset key
    pair_labels = []

    for idx in trainset_iemocap.keys:
        lable_tem = trainset_iemocap.videoLabels[idx]
        for i in range(0, len(lable_tem)-(emotion_pair_size+1), 1):
            pair_labels.append([lable_tem[i+j] for j in range(emotion_pair_size+1)])
    
    emotion_count = [0]*6
    # probability of a target label occuring for a given emotion pair
    probability = {}
    total_labels = 0
    
    # Number of occurence of emotion in each emoitons sequnce: pair_labels
    emotion_sequence_count = {} 
    
    for sequence in pair_labels:
        sequence_str = ''.join(map(str,sequence[:-1]))
        if sequence_str not in emotion_sequence_count:
            emotion_sequence_count[sequence_str]=[0]*6
        emotion_sequence_count[sequence_str][sequence[-1]] += 1    
    for sequence, counts in emotion_sequence_count.items():
        total_emotions = sum(counts)
        probability[sequence] = [i/total_emotions for i in counts]

                                             
    knowledge_pair = pd.DataFrame(probability.items(),columns = ['pair_Labels', 'Probability'])
    print(knowledge_pair)
            
    knowledge_pair.index = pd.Series(knowledge_pair.pair_Labels)
    knowledge_pair.to_pickle('knowledge_pair.pkl')
