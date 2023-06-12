import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.nn.functional as F

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import argparse
import pandas as pd
from tqdm import tqdm
from collections import namedtuple, deque
import random

from dueling_dqn_model import DuelingDQN

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support, recall_score, precision_score

from pair_dataloader import IEMOCAP_pair_Dataset

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

loss_weights = torch.FloatTensor([
                                    1/0.086747,
                                    1/0.144406,
                                    1/0.227883,
                                    1/0.160585,
                                    1/0.127711,
                                    1/0.252668,
                                    ])
torch.set_default_tensor_type(torch.FloatTensor)

# CUDA variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

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
    parser.add_argument('--attribute', type=int, default=1, help='AVEC attribute')
    args = parser.parse_args()

    print(args)

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

    sum_iemocap = 0
    
    criterion = nn.CrossEntropyLoss(loss_weights.cuda() if cuda else loss_weights)

    test_label_pair_iemocap = []
    test_videoText_pair_iemocap = []
    test_videoAudio_pair_iemocap = []
    test_videoVisual_pair_iemocap = []

    epoch_num = 5
    batch = 10
    LEARNING_RATE = 0.0001
    ALPHA = 0.95
    EPS = 0.01
    exploration=LinearSchedule(1000000, 0.1)
    gamma = 0.9 #0.99
    target_update_freq = 100
    learning_freq = 4
    double_dqn = True
    num_param_updates = 0
    greedy = 0.95
    mu = 0
    sigma = 0.5

    cuda = 0
    device = torch.device("cuda:%d" % cuda if torch.cuda.is_available() else "cpu")

    ########################
    Q = DuelingDQN(learning_rate = LEARNING_RATE,batch_size = batch)
    Q_target = DuelingDQN(learning_rate = LEARNING_RATE,batch_size = batch)
    ########################

    Q.to(device)
    Q_target.to(device)

    optimizer = OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(lr=LEARNING_RATE, eps=EPS, weight_decay=0.00001)
    )
    optimizer = optimizer.constructor(Q.parameters(), **optimizer.kwargs)

    ##############################begin train#####################################################
    pair_env_train = IEMOCAP_pair_Dataset(path='trainset_pair.pkl')
    train_loader = DataLoader(
                                dataset=pair_env_train,
                                batch_size=batch,
                                shuffle=True,
                                num_workers=0,
    )
    pair_env_test = IEMOCAP_pair_Dataset(path='testset_pair.pkl')
    test_loader = DataLoader(
                                dataset=pair_env_test,
                                batch_size=10,
                                shuffle=False,
                                num_workers=0,
    )
    domain = pd.read_pickle('knowledge_pair.pkl')
    title = domain.pair_Labels
    p = domain.Probability
    
    acc_best = 0
    for epoch in tqdm(range(epoch_num)):
        print("now the epoch number is: %d" % epoch)
        tot_num = 0
        tot_right = 0

        # with grad
        for i, data in enumerate(train_loader):
            states_titles, pair_Labels, states_f_text, states_f_audio, states_f_visual, next_states_titles, next_states_f_text,\
            next_states_f_audio, next_states_f_visual, action, done = data
            
            observations = data
            batch_tem = len(action)
            done = done.squeeze()
            
            pair_Labels, states_f_text, states_f_audio, states_f_visual,next_states_f_text, next_states_f_audio, next_states_f_visual, action, done = pair_Labels.to(device), states_f_text.to(device), states_f_audio.to(device), states_f_visual.to(device),next_states_f_text.to(device), next_states_f_audio.to(device),\
            next_states_f_visual.to(device), action.to(device), done.to(device)
            
            Q.train()
            
            # Without Exploration, use the following computation:
#             q_values = Q(states_f_text,states_f_audio,states_f_visual)
#             q_action = torch.argmax(q_values, dim = 1) # for dqn
            

            # With exploration, use the following computation:
            sample = random.random()
            threshold = exploration.value(i)
            q_values = Q(states_f_text,states_f_audio,states_f_visual)
            if np.random.uniform() > greedy:
                q_action = torch.argmax(q_values, dim = 1)
                
                # From old version 
                q_action = F.log_softmax(q_values, dim = 1)
                q_action_t = F.softmax(q_values, dim = 1)
                q_action = torch.argmax(q_action, dim = 1)
                
            else:
                q_action = torch.LongTensor(np.random.randint(0,6,size = 10))
                q_action = q_action.to(device)

            ##############################################################################dqn
            q_s_a = q_values.gather(1, q_action.unsqueeze(1))
            q_s_a = q_s_a.squeeze()

            reward = []
        
            for j in range(batch_tem):
                if q_action[j] == action[j]:
                    reward.extend([2.0])
                    tot_right +=1
                else:
                    reward.extend([-2.0])
            tot_num += batch_tem

            reward = np.array(reward)
            
            reward = torch.from_numpy(reward)
            reward = reward.to(device)

            if (double_dqn):
                # ---------------
                #   double DQN
                # ---------------
                q_tp1_values = Q(next_states_f_text,next_states_f_audio,next_states_f_visual).detach()
                _, a_prime = q_tp1_values.max(1)

                q_target_tp1_values = Q_target(next_states_f_text,
                                               next_states_f_audio,
                                               next_states_f_visual).detach()
                q_target_s_a_prime = q_target_tp1_values.gather(1, a_prime.unsqueeze(1))
                q_target_s_a_prime = q_target_s_a_prime.squeeze()

                # if current state is end of episode, then there is no next Q value
                q_target_s_a_prime = (1 - done) * q_target_s_a_prime 
                # Compute Bellman error
                expected_Q = reward + gamma * q_target_s_a_prime
                
            else:
                # ---------------
                #   regular DQN
                # ---------------
                q_tp1_values = Q_target(next_states_f_text,
                                        next_states_f_audio,next_states_f_visual).detach()
                q_s_a_prime, a_prime = q_tp1_values.max(1)
                
                # if current state is end of episode, then there is no next Q value
                q_s_a_prime = (1 - done) * q_s_a_prime 

                # Compute Bellman error = r + gamma * Q(s',a', theta_i_frozen) - Q(s, a, theta_i)
                expected_Q = reward + gamma * q_s_a_prime

            # backwards pass
            optimizer.zero_grad()
            q_s_a = q_s_a.type(torch.float64)
            error = expected_Q - q_s_a

            clipped_error = -1.0 * error
            q_s_a.backward(clipped_error)
            ##############################################################################dqn

            # update
            optimizer.step()
            num_param_updates += 1

            # update target Q network weights with current Q network weights
            if num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())
        
        print("now the epoch number is: %d" % epoch)
        tot_num_t = 0.0
        tot_right_t = 0


        tot = [0] * 6
        tot_r = [0] * 6
        acc = [0] * 6
        
        try_tot_revise = []
        try_tot_revise_real = []
        try_tot_real = []
        try_tot = []
        Y_valid = []
        action_r = []
        tot_right_try = 0
        tot_right_try_revise = 0
        tot_right_revise_max_real = 0
        tot_right_revise_real = 0

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                states_titles, pair_Labels, states_f_text, states_f_audio, states_f_visual, next_states_titles, next_states_f_text, next_states_f_audio, next_states_f_visual, action, done = data
                observations = data
                done = done.squeeze()
                batch_tem = len(action)

                pair_Labels, states_f_text, states_f_audio, states_f_visual, next_states_f_text, next_states_f_audio, next_states_f_visual, action, done = pair_Labels.to(
                    device), states_f_text.to(device), states_f_audio.to(device), states_f_visual.to(device), next_states_f_text.to(device), next_states_f_audio.to(
                        device), next_states_f_visual.to(device), action.to(device), done.to(device)
                
                if states_f_visual.size()[0] == 1:
                    continue
                
                Q.eval()
                q_values = Q(states_f_text,states_f_audio,states_f_visual)
                q_action = torch.argmax(q_values, dim = 1) # for dqn

                q_action_t = F.softmax(q_values, dim = 1)

                pair_Labels = torch.squeeze(pair_Labels) 


                action = action.squeeze()
                action_r.extend(action)
                tot_num_t += len(action)
                # tot_num += batch_tem*4

                for j in range(batch_tem):
                    tem = q_values[j]
                    Y_valid.append(action[j])



                    tem_pair_real = pair_Labels[j]
                    t_0_real = tem_pair_real[0].tolist()
                    t_1_real = tem_pair_real[1].tolist()
                    t_2_real = tem_pair_real[2].tolist()

                    t_real = str(t_0_real)+str(t_1_real)+str(t_2_real)
                    if t_real in title:
                        t_t_real = p[t_real]
                    else:
                        t_t_real = torch.ones(6)
                    t_t_real = torch.tensor(t_t_real)
                    t_t_real = torch.squeeze(t_t_real)
                    t_t_real = t_t_real.to(device)
                    t_max_real = torch.tensor(t_t_real)
                    t_max_real = torch.squeeze(t_max_real)
                    t_max_real = torch.argmax(t_max_real, dim = 0)
                    try_tot_real.append(t_max_real)
                    tem_t = q_action_t[j]

                    t_revise_real = tem_t + 1.5*t_t_real
                    t_revise_real = torch.argmax(t_revise_real, dim = 0)
                    try_tot_revise_real.append(t_revise_real)

                
            for prediction, label in zip(try_tot_revise_real, action_r):
                if prediction == label:
                    tot_right_t +=1
                    tot_r[label] +=1
                tot[prediction] +=1
                acc[label] +=1
            
            true = (np.array(action_r)).astype(int)
            pred = (np.array(try_tot_revise_real)).astype(int)
            
            # Overall accuracy
            acc_total = accuracy_score(true, pred)
            
            # Overall recall and recall for each emotion label
            recall = tot_right_t/tot_num_t
            recall_list= [0] * 6
            for label in range(len(recall_list)):
                recall_list[label] = recall_score(true, pred, labels = [label],average='micro')
                
            # Over precision and precision fro each emoition category
            precision = tot_right/tot_num
            precision_list = [0] * 6
            for label in range(len(precision_list)):
                precision_list[label] = precision_score(true, pred, labels = [label],average='macro')
            
            # Over all F1 score and F1 score for each emotion category
            F_list = [0] * 6
            F_total = f1_score(true, pred, average='weighted')
            for label in range(len(F_list)):
                F_list[label] = f1_score(true, pred, labels = [label], average='weighted')

            Y_valid = torch.tensor(Y_valid)
            Y_valid = torch.squeeze(Y_valid)
 
            try_tot_revise_real = torch.tensor(try_tot_revise_real)
            tot_right_revise_real += torch.sum(torch.eq(Y_valid, try_tot_revise_real)) #real pair
            try_tot_real = torch.tensor(try_tot_real)
            tot_right_revise_max_real += torch.sum(torch.eq(Y_valid, try_tot_real)) #library

            acc_r_t = tot_right_t/tot_num_t # own

            print("tot_right_revise_real:",tot_right_revise_real)
            print("tot_num_t:",tot_num_t)
            print("tot_right_revise_max_real:",tot_right_revise_max_real)
            
            acc_real = tot_right_revise_real/tot_num_t
            acc_real_max = tot_right_revise_max_real/tot_num_t
            if acc_r_t > acc_best:

                acc_best = acc_r_t
            print('test:', epoch + 1, i + 1,'acc_real:',acc_r_t, 'acc_real_best:', acc_best,
                  'acc_recognition:', acc_real_max)

            print('test_real', epoch + 1, i + 1, 'recall:',recall_list)
            
            print('test_real',epoch + 1, i + 1,'precision:', precision_list)
            
            print('test_real:',epoch + 1, i + 1,'F:',F_total, 'F_labels:',F_list)
        
        acc = tot_right/tot_num
        print('train [%d, %5d] acc: %.3f' % (epoch + 1, i + 1, (tot_right/tot_num)))
    
    torch.save(Q.state_dict(), 'Q.pkl')
    
    ##############################end train#####################################################
