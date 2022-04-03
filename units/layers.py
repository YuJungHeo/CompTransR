import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.autograd import Variable

"""
This code is based on Thomas Kipf's repository.
https://github.com/ethanfetaya/NRI
"""

class FC(nn.Module):
    """One-layer fully-connected ELU net with batch norm."""
    def __init__(self, n_in, n_out, do_prob=0.):
        super(FC, self).__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        x = F.elu(self.fc(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        return self.batch_norm(x)
    
class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)
    
class RELEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out_node, n_out_edge, n_subspace, do_prob=0., factor=True):
        super(RELEncoder, self).__init__()
        self.h_dim = n_hid
        self.n_subspace = n_subspace

        self.head_enc = MLP(n_in, n_hid, n_hid, do_prob)
        self.tail_enc = MLP(n_in, n_hid, n_hid, do_prob)
        self.union_enc = MLP(n_in, n_hid, n_hid, do_prob)

        self.msg_enc = nn.ModuleList([MLP(n_hid, n_hid, n_hid, do_prob) for _ in range(n_subspace)])
        self.pred_enc = nn.ModuleList([MLP(n_out_edge, n_hid, n_hid, do_prob) for _ in range(n_subspace)])
        self.edge_readout = nn.ModuleList([nn.Linear(n_hid*2, n_out_edge) for _ in range(n_subspace)])

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        rel_rec = torch.transpose(rel_rec, 2, 1)
        incoming = torch.matmul(rel_rec, x)  # [*B, n, #odiag]*[B, #odiag, h]=[B, n, h]
        return incoming / incoming.size(1)

    def node2edge(self, h, t, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, t)  # [*B, #odiag, n]*[B, n, h]=[B, #odiag, h]
        senders = torch.matmul(rel_send, h)  # [*B, #odiag, n]*[B, n, h]=[B, #odiag, h]
        edges = torch.cat([receivers, senders], dim=2)  # [B, #odiag, 2*h]
        return edges

    def n2e_trans(self, h, t, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, t)  # [*B, #odiag, n]*[B, n, h]=[B, #odiag, h]
        senders = torch.matmul(rel_send, h)  # [*B, #odiag, n]*[B, n, h]=[B, #odiag, h]
        edges = receivers - senders  # [B, #odiag, h]
        return edges, senders, receivers

    def e_update(self, msgs, gamma):
        # msgs: [B, #odiag, h*#subs], gamma: [B, #odiag, #subs]
        msgs_att = torch.einsum('bik,bijk->bijk', (gamma, msgs))
        msgs_att = torch.sum(msgs_att, -1)  # [B, #odiag, h]
        return msgs_att

    def forward(self, node, edge, rels, rel_rec, rel_send, loss_mode):
        head = self.head_enc(node)
        tail = self.tail_enc(node)
        union = self.union_enc(edge)
        n_odiag = union.shape[1]

        for j in range(self.n_subspace):
            enc_idx = j
            h_proj = self.msg_enc[enc_idx](head)  # [B, n, h]
            t_proj = self.msg_enc[enc_idx](tail)  # [B, n, h]
            rel2rel = self.pred_enc[enc_idx](rels)

            ht2rel, h_proj, t_proj = self.n2e_trans(h_proj, t_proj, rel_rec, rel_send)  # [B, #odiag, h]
            e_enc = torch.cat((ht2rel, union), dim=-1)
            e_enc = e_enc[:, :, :, None]
            ht2rel = ht2rel[:, :, :, None]
            rel2rel = rel2rel[:, :, :, None]

            if j == 0:
                trans = e_enc
                ht2rels = ht2rel
                rel2rels = rel2rel
            else:
                trans = torch.cat((trans, e_enc), dim=-1)  # [B, #odiag, h, #subs]
                ht2rels = torch.cat((ht2rels, ht2rel), dim=-1)
                rel2rels = torch.cat((rel2rels, rel2rel), dim=-1)

        for k in range(self.n_subspace):
            e_tmp = self.edge_readout[k](trans[:, :, :, k])  # [B, #odiag, #out]
            e_tmp[:, :, 0] = -float("inf")

            if loss_mode == 'bce' or loss_mode == 'margin':
                logits = torch.sigmoid(e_tmp)
            elif loss_mode == 'ce':
                logits = F.log_softmax(e_tmp, dim=-1)

            logits = logits[:, :, :, None]
            if k == 0:
                e_out = logits
            else:
                e_out = torch.cat((e_out, logits), dim=-1)  # [B, #odiag, 51, #subs]

        return e_out, ht2rels, rel2rels
    
class ClassEmbedding(Module):
    def __init__(self, ntoken, emb_dim, dropout, trainable=True):
        super(ClassEmbedding, self).__init__()
        #self.emb = nn.Embedding(ntoken, emb_dim, padding_idx=0)
        self.emb = torch.nn.Linear(ntoken, emb_dim, bias=False)
        self.emb.weight.requires_grad = trainable
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data = weight_init.transpose(0, 1)

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb
    
class NonLinear(Module):
    def __init__(self, feat_dim, emb_dim):
        super(NonLinear, self).__init__()
        self.l1 = torch.nn.Linear(feat_dim, emb_dim, bias=True)
        self.l2 = torch.nn.ReLU()
        
    def forward(self, node):
        out = self.l2(self.l1(node))
        return out