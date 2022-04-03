import os
import torch
import pickle
import numpy as np
import random

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def make_one_hot(labels, num_labels):
    '''
    Converts an integer label to a one-hot Variable
    Args
        labels : [B*N], each value is an integer representing correct classification
        num_labels : num of true labels
    Returns
        target : [B*N*num_labels], one-hot encoded
    '''
    onehot = torch.eye(num_labels)
    out = onehot[labels.long()]
    return out.type(torch.cuda.FloatTensor)

def make_union_one_hot(labels, num_labels):
    onehot = torch.eye(num_labels)
    sub_onehot = onehot[labels[:,:,0].long()]
    obj_onehot = onehot[labels[:,:,1].long()]*(-1)
    out = sub_onehot.type(torch.IntTensor)|obj_onehot.type(torch.IntTensor)
    return out.type(torch.cuda.FloatTensor)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def mask1d_2dmat(length, target, val=0):
    mask = torch.arange(max(length)).expand(len(length), max(length)) >= length.unsqueeze(1)
    target.masked_fill_(mask.cuda().data.bool(), val)
    return target

def mask1d_3dmat(length, target, val=0):
    mask = torch.arange(max(length)).expand(len(length), max(length)) >= length.unsqueeze(1)
    mask = mask.unsqueeze(2).cuda()
    target.masked_fill_(mask.data.bool(), val)
    return target

def mask1d_4dmat(length, target, val=0):
    mask = torch.arange(max(length)).expand(len(length), max(length)) >= length.unsqueeze(1)
    mask = mask.unsqueeze(2).unsqueeze(3).cuda()
    target.masked_fill_(mask.data.bool(), val)
    return target

def mask2d_3dmat(length, target, val=0):
    mask = torch.arange(max(length)).expand(len(length), max(length)) >= length.unsqueeze(1)
    mask_sub = mask.unsqueeze(2).cuda()
    mask_obj = mask.unsqueeze(1).cuda()
    
    mask_diag = np.zeros([max(length), max(length)]) + np.eye(max(length))
    mask_diag = torch.from_numpy(mask_diag).cuda()
    
    target.masked_fill_(mask_sub.data.bool(), val)
    target.masked_fill_(mask_obj.data.bool(), val)
    target.masked_fill_(mask_diag.data.bool(), val)
    return target

def mask2d_4dmat(length, target, val=0):
    mask = torch.arange(max(length)).expand(len(length), max(length)) >= length.unsqueeze(1)
    mask_sub = mask.unsqueeze(1).unsqueeze(3).cuda()
    mask_obj = mask.unsqueeze(1).unsqueeze(2).cuda()
    
    mask_diag = np.zeros([max(length), max(length)]) + np.eye(max(length))
    mask_diag = torch.from_numpy(mask_diag).unsqueeze(0).unsqueeze(1).cuda()
    
    target.masked_fill_(mask_sub.data.bool(), val)
    target.masked_fill_(mask_obj.data.bool(), val)
    target.masked_fill_(mask_diag.data.bool(), val)
    return target

def mask2d_5dmat(length, target, val=0):
    mask = torch.arange(max(length)).expand(len(length), max(length)) >= length.unsqueeze(1)
    mask_sub = mask.unsqueeze(2).unsqueeze(3).unsqueeze(4).cuda()
    mask_obj = mask.unsqueeze(1).unsqueeze(3).unsqueeze(4).cuda()
    
    mask_diag = np.zeros([max(length), max(length)]) + np.eye(max(length))
    mask_diag = torch.from_numpy(mask_diag).unsqueeze(0).unsqueeze(3).unsqueeze(4).cuda()
    
    target.masked_fill_(mask_sub.data.bool(), val)
    target.masked_fill_(mask_obj.data.bool(), val)
    target.masked_fill_(mask_diag.data.bool(), val)
    return target

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word
        if word2idx == {} and idx2word == []:
            self.idx2word.append('none')
            self.word2idx['none'] = 0
        
    @property
    def ntoken(self):
        return len(self.word2idx)

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word)-1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)