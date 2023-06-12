import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import tensorflow as tf
from GNN import GNN

def make_graph(features, num_batches, device):
    '''
    Input: features (size = num_batches,4,512*2)
    return: 
    '''
    node_features = []
    edge_index = []
    edge_type = []
    batch_size = features.size(0)
    length_sum = 0
    pair_size = features.size(1)
    for i in range(num_batches):
        node_features.append(features[i, :, :])
        for j in range(pair_size):
            for k in range(pair_size):
                edge_index.append(torch.tensor([
                    j + length_sum,
                    k + length_sum]))
                edge_type.append(j * pair_size + k)
        length_sum += pair_size
    node_features = torch.cat(node_features, dim=0).to(device)
    edge_index = torch.stack(edge_index).t().contiguous().to(device)
    edge_type = torch.tensor(edge_type).long().to(device)
    return (node_features, edge_index, edge_type)


class DuelingDQN(nn.Module):
    def __init__(
        self, 
        n_actions = 6,
        n_features = 100, 
        learning_rate = 0.001, 
        reward_decay = 0.9, 
        e_greedy = 0.9, 
        replace_target_iter = 200, 
        memory_size = 500, 
        batch_size = 32, 
        e_greedy_increment = None, 
        output_graph = False, 
        dueling = True, 
        sess = None, 
        gnn_num_layers = 1024,
    ):
        super(DuelingDQN, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.dropProb = 0.4
        self.num_layer = 1
        
        self.dueling = dueling
        
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 4 * 6 + 2)) # 2states,3modals,4timestamps
        
        self.norm_uni_4d = nn.BatchNorm2d(n_features, track_running_stats = False)
        self.norm_uni_v_4d = nn.BatchNorm2d(512, track_running_stats = False)
        self.drop = nn.Dropout(p = self.dropProb)
        self.line_uni = nn.Linear(n_features, 512)
        self.line_uni_v = nn.Linear(512, 512)
        self.line_bi = nn.Linear((512*2*2), 512)
        self.line_mul = nn.Linear((512*2), 512*4)
        self.norm_mul = nn.BatchNorm1d(512*4)
        
        self.gcn = GNN(gnn_num_layers, gnn_num_layers, gnn_num_layers)
        
        self.gru_uni = nn.GRU(512, 512, num_layers=self.num_layer, dropout=self.dropProb,
                              bidirectional=True, batch_first=True)
        self.gru_mul = nn.GRU(512, 512, num_layers=self.num_layer, dropout=self.dropProb,
                              bidirectional=True, batch_first=True)
        
        self.fc1_adv = nn.Linear(in_features=4*512*2, out_features=512)
        self.fc1_val = nn.Linear(in_features=4*512*2, out_features=512)
        
        self.fc2_adv = nn.Linear(in_features=512, out_features=self.n_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)
        
        self.relu = nn.ReLU()
        self.att = nn.Linear(512*2, 1, bias=False)

    
    def forward(self, states_t, states_a, states_v):
        t = states_v.size()
        batch = states_t.size()[0]
#         print('states_t.size()',states_t.size())
        seq = states_t.size()[2]
        cuda = 0
        device = torch.device('cuda:%d' % cuda if torch.cuda.is_available() else 'cpu')
#         batch_l = torch.tensor(batch)
#         batch_l = batch_l.to(device)
        
        def process_state(states, is_video ):
#             print('states before',states.size())
            states = states.data.contiguous().view(batch, -1, 1, seq)
#             print('states',states.size())
            if is_video:
                states = self.norm_uni_v_4d(states)
            else:
                states = self.norm_uni_4d(states)
            states = torch.squeeze(states)
            states = states.data.contiguous().view(batch, seq, -1)
            if is_video:
                states_in = self.line_uni_v(self.drop(states))
            else:
                states_in = self.line_uni(self.drop(states))
            (states_out, _) = self.gru_uni(states_in)
            states_out_M = states_out.permute(1, 0, 2)
            states_out_M = self.att(states_out_M)
            states_Selector = F.softmax(states_out_M, dim=0).permute(1, 2, 0)
            states_State = torch.matmul(states_Selector, states_out).squeeze()
            return states_State

        states_t_State = process_state(states_t, False)
        states_a_State = process_state(states_a, False)
        states_v_State = process_state(states_v, True)
        states_at = self.line_bi(torch.cat([states_a_State,states_t_State], 1))
        states_vt = self.line_bi(torch.cat([states_v_State,states_t_State], 1))
        states_avt = self.line_mul(torch.cat([states_at,states_vt], 1))
        states_avt = self.drop(self.norm_mul(states_avt))
        states_avt = states_avt.data.contiguous().view(batch, -1, 512)
        (output, _) = self.gru_mul(states_avt)
        (features, edge_index, edge_type) = make_graph(output, batch, device)
        graph_out = self.gcn(features, edge_index, edge_type)
        output = graph_out.data.contiguous().view(batch, -1)
        adv = self.relu(self.fc1_adv(output))
        adv = self.fc2_adv(adv)
        val = self.relu(self.fc1_val(output))
        val = self.fc2_val(val).expand(output.size(0), self.n_actions)
        output = val + adv - adv.mean(1).unsqueeze(1).expand(output.size(0), self.n_actions)
        return output
