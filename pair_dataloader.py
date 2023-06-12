import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd

'states_titles', 'pair_Labels', 'states_f_text', 'states_f_audio', 'states_f_visual', 
'next_states_titles', 'next_states_f_text', 'next_states_f_audio', 'next_states_f_visual', 
'action', 'done'

class IEMOCAP_pair_Dataset(Dataset):

    def __init__(self, path):
        pair_env = pd.read_pickle(path)
        
        self.states_titles, self.pair_Labels, self.states_f_text, self.states_f_audio,\
        self.states_f_visual, self.next_states_titles, self.next_states_f_text, self.next_states_f_audio, self.next_states_f_visual,\
        self.action, self.done = pair_env.states_titles, pair_env.pair_Labels, pair_env.states_f_text, pair_env.states_f_audio,\
        pair_env.states_f_visual, pair_env.next_states_titles, pair_env.next_states_f_text, pair_env.next_states_f_audio, pair_env.next_states_f_visual,\
        pair_env.action, pair_env.done
        '''
        action index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = [x for x in (self.states_titles)]
#         print('keys', self.keys)
        self.len = len(self.keys)
#         print('item', self.len)

    def __getitem__(self, index):
        # title = self.keys[index]
        title = index
#         print('type',self.states_titles[title],\
#                type(self.action[title]),\
#                type(self.done[title]))
#         print('item',self.states_titles[title],\
#                torch.LongTensor(self.action[title]).size(),\
#                torch.LongTensor(self.done[title]).size())
#         print('sizes',self.states_titles[title],\
#                self.action[title],\
#                self.done[title])
        return self.states_titles[title],\
               torch.LongTensor(self.pair_Labels[title]),\
               torch.FloatTensor(self.states_f_text[title]),\
               torch.FloatTensor(self.states_f_audio[title]),\
               torch.FloatTensor(self.states_f_visual[title]),\
               self.next_states_titles[title],\
               torch.FloatTensor(self.next_states_f_text[title]),\
               torch.FloatTensor(self.next_states_f_audio[title]),\
               torch.FloatTensor(self.next_states_f_visual[title]),\
               torch.LongTensor([self.action[title]]),\
               torch.LongTensor([self.done[title]])
               

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        print('data',data)
        dat = pd.DataFrame(data)
#         print('dat',dat)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

