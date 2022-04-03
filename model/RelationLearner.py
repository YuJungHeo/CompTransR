import numpy as np
import os
import torch
import torch.nn as nn
from utils import *
from units.layers import *

class RelLearner(nn.Module):

    def __init__(self, conf):
        super(RelLearner, self).__init__()
        self.num_gpus = conf.ngpu
        self.batch_size = conf.batch_size
        self.dropout = conf.dropout
        self.n_subspace = conf.n_subspace
        self.mtr_mode = conf.mtr_mode
        self.is_gtadj = conf.is_gtadj

        self.vdim_init = 2048
        self.v_dim = 512
        self.p_dim = 4
        self.s_dim = conf.emb_dim

        if conf.dataset == 'vg':
            self.o_dim_node = 151
            self.o_dim_edge = 51
            self.h_dim = 256
        elif conf.dataset == 'vrd':
            self.o_dim_node = 100
            self.o_dim_edge = 71
            self.h_dim = 256
        elif conf.dataset == 'vrr-vg':
            self.o_dim_node = 1600
            self.o_dim_edge = 117
            self.h_dim = 256

        self.factor = True
        self.loss_mode = conf.loss_mode

        if self.mtr_mode == 'preddet':
            self.cemb_node = ClassEmbedding(self.o_dim_node, conf.emb_dim, 0.0)
            self.cemb_node.init_embedding(os.path.join(conf.DATA_PATH, 'glove6b_objinit_%dd.npy' % self.s_dim))
            self.linear_vn = NonLinear(2048, self.v_dim)
            self.linear_ve = NonLinear(2048, self.v_dim)
        
        self.encoder = RELEncoder(self.v_dim + self.p_dim + self.s_dim, self.h_dim,
                                  self.o_dim_node, self.o_dim_edge, self.n_subspace, 
                                  self.dropout, self.factor)
        
    def forward(self, im_sizes, image_offset, gt_boxes, gt_classes, vnode, vedge, pnode,
                pedge, snode, sedge, gt_adjmat, gt_rel, gt_nnodes, kg_prior):
        
        variable_batch = vnode.shape[0]
        n_node = vnode.shape[1]
    
        triu_mask = np.mask_indices(n_node, np.triu, 1)
        off_diag_2d = np.ones([n_node, n_node]) - np.eye(n_node)
        off_diag = np.tile(off_diag_2d, (variable_batch, 1, 1))
        idxs = np.where(off_diag)
        
        if self.mtr_mode == 'preddet':
            vnode = self.linear_vn(vnode)
            snode = make_one_hot(snode, self.o_dim_node)
            snode = self.cemb_node(snode)

            vedge_init = torch.zeros([variable_batch, n_node, n_node, self.vdim_init]).cuda()
            vedge_init[:, triu_mask[0], triu_mask[1]] = vedge
            vedge_init[:, triu_mask[1], triu_mask[0]] = vedge

            vedge = vedge_init[idxs[0], idxs[1], idxs[2], :]
            vedge = vedge.reshape([variable_batch, -1, self.vdim_init])
            vedge = self.linear_ve(vedge)

            pedge_init = torch.zeros([variable_batch, n_node, n_node, self.p_dim]).cuda()
            pedge_init[:, triu_mask[0], triu_mask[1]] = pedge
            pedge_init[:, triu_mask[1], triu_mask[0]] = -pedge
            pedge = pedge_init[idxs[0], idxs[1], idxs[2], :]
            pedge = pedge.reshape([variable_batch, -1, self.p_dim])

            sedge = make_union_one_hot(sedge, self.o_dim_node)
            sedge = self.cemb_node(sedge)
            sedge_init = torch.zeros([variable_batch, n_node, n_node, self.s_dim]).cuda()
            sedge_init[:, triu_mask[0], triu_mask[1]] = sedge
            sedge_init[:, triu_mask[1], triu_mask[0]] = -sedge
            sedge = sedge_init[idxs[0], idxs[1], idxs[2], :]
            sedge = sedge.reshape([variable_batch, -1, self.s_dim])

            adjmat_f = gt_adjmat > 0

            rels = gt_adjmat[idxs[0], idxs[1], idxs[2]]
            rels = rels.reshape([variable_batch, -1])
            rels = make_one_hot(rels, self.o_dim_edge)

            node_f = torch.cat((vnode, pnode, snode), dim=-1)
            edge_f = torch.cat((vedge, pedge, sedge), dim=-1)

        rel_rec = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag_2d)[1]), dtype=np.float32)).cuda()
        rel_send = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag_2d)[0]), dtype=np.float32)).cuda()
        pred_edge_odiag, ht2rels_odiag, rel2rels_odiag = self.encoder(node_f, edge_f, rels, rel_rec, rel_send, self.loss_mode)

        pred_edge = torch.zeros([variable_batch, n_node, n_node, self.o_dim_edge, self.n_subspace]).cuda()
        ht2rels = torch.zeros([variable_batch, n_node, n_node, self.h_dim, self.n_subspace]).cuda()
        rel2rels = torch.zeros([variable_batch, n_node, n_node, self.h_dim, self.n_subspace]).cuda()

        for i in range(self.n_subspace):
            pred_edge[:, np.where(off_diag_2d)[0], np.where(off_diag_2d)[1], :, i] = pred_edge_odiag[:,:,:,i]
            ht2rels[:, np.where(off_diag_2d)[0], np.where(off_diag_2d)[1], :, i] = ht2rels_odiag[:,:,:,i]
            rel2rels[:, np.where(off_diag_2d)[0], np.where(off_diag_2d)[1], :, i] = rel2rels_odiag[:,:,:,i]
        
        pred_edge = mask2d_5dmat(gt_nnodes, pred_edge)
        pred_edge = pred_edge.permute(0, 3, 1, 2, 4)
        ht2rels = mask2d_5dmat(gt_nnodes, ht2rels)
        rel2rels = mask2d_5dmat(gt_nnodes, rel2rels)
        
        torch.torch.cuda.empty_cache()
        return pred_edge, ht2rels, rel2rels

    def __getitem__(self, batch):
        batch.scatter()
        output_device = 0
        
        if self.num_gpus == 1:
            return self(*batch[0])
        
        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])
    
        return nn.parallel.gather(outputs, output_device)