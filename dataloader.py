import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd

class IEMOCAPDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
#         print('item', self.videoIDs)
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        # self.keys = [x for x in (self.videoIDs)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
#         print('item', torch.FloatTensor(self.videoText[vid]),\
#                torch.FloatTensor(self.videoVisual[vid]),\
#                torch.FloatTensor(self.videoAudio[vid]),\
#                torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
#                                   self.videoSpeakers[vid]]),\
#                torch.FloatTensor([1]*len(self.videoLabels[vid])),\
#                torch.LongTensor(self.videoLabels[vid]),\
#                vid)
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid
    

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
#         print('dat' , dat)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]
