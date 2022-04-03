import os
import torch
import json
import math
import h5py
import numpy as np
import pickle
import collections
from PIL import Image
from torch.utils import data
import torch.nn.functional as F
from dataloaders.blob import Blob

import random
from torch.utils.data.sampler import Sampler


class Dataset(data.Dataset):

    def __init__(self, conf, mode):
        self.conf = conf
        self.mode = mode

        if self.conf.mode == 'train':
            self.is_training = True
        else:
            self.is_training = False

        self.gt_classes, self.gt_relationships, self.gt_boxes, self.gt_nboxes, self.gt_adjmats, self.img_size, self.img_idxs = load_graphs_vrd(
            self.mode, self.conf.DATA_PATH)

        self.kg_priors = knowledge_prior(self.conf.DATA_PATH, self.gt_classes, self.img_idxs)
        self.cluster_indices = make_cluster_indices(self.gt_nboxes, self.conf.batch_size, self.mode)
        self.num_batch = len(self.cluster_indices)

    def __len__(self):
        return len(self.img_idxs)

    def __getitem__(self, idx):
        # print (idx)
        if np.isnan(idx):
            return
        else:
            idx = int(idx)

        img_idx = self.img_idxs[idx]
        vnode = np.load(os.path.join(self.conf.DATA_PATH + '/VRD%s_node_%s/%s.npy' % (self.conf.bmode, self.mode, img_idx)))
        vedge = np.load(os.path.join(self.conf.DATA_PATH + '/VRD%s_edge_%s/%s.npy' % (self.conf.bmode, self.mode, img_idx)))

        gt_boxes = self.gt_boxes[idx].copy()
        nnode = gt_boxes.shape[0]
        gt_rels = self.gt_relationships[idx].copy()
        gt_classes = self.gt_classes[idx].copy()
        im_size = self.img_size[idx]
        kg_priors = self.kg_priors[idx].copy()
        all_pairs = np.transpose(np.mask_indices(nnode, np.triu, 1))

        if self.conf.p_type == 'coord':
            pnode = gt_boxes
            pedge = obj_union_position_coord(pnode, all_pairs)

        elif self.conf.p_type == 'mask':
            pnode = obj_position_mask(im_size, gt_boxes)
            pedge = obj_union_position_mask(pnode, all_pairs)

        if self.conf.mtr_mode == 'predcls' or 'preddet':
            snode = gt_classes
            sedge = obj_union_semantic_feature(snode, all_pairs)

        gt_adjmat = self.gt_adjmats[idx].copy()

        entry = {
            'img_size': im_size,
            'gt_boxes': gt_boxes,
            'gt_classes': gt_classes,
            'gt_nnodes': nnode,
            'gt_relations': gt_rels,
            'gt_adjmat': gt_adjmat,
            'kg_priors': kg_priors,
            'index': idx,
            'vnode': vnode,
            'vedge': vedge,
            'pnode': pnode,
            'pedge': pedge,
            'snode': snode,
            'sedge': sedge
        }

        assertion_checks(entry)
        return entry


def obj_position_mask(im_size, boxes, mask_dim=32):
    """
    args
        im_size: (ih, iw) of img
        boxes: #boxes * [xmin, ymin, xmax, ymax]
    return
        masks: ndarray of masks for all object boxes: (#boxes * mask_dim * mask_dim)
    """

    ih = im_size[0]
    iw = im_size[1]
    rh = mask_dim / ih
    rw = mask_dim / iw
    n_boxes = boxes.shape[0]
    masks = np.zeros((n_boxes, mask_dim, mask_dim))
    for i in range(n_boxes):
        x1 = max(0, int(math.floor(boxes[i][0] * rw)))
        x2 = min(mask_dim, int(math.ceil(boxes[i][2] * rw)))
        y1 = max(0, int(math.floor(boxes[i][1] * rh)))
        y2 = min(mask_dim, int(math.ceil(boxes[i][3] * rh)))

        masks[i, y1: y2, x1: x2] = 1
        assert (masks[i].sum() == (y2 - y1) * (x2 - x1))
    return masks


def obj_union_position_mask(masks, pair_inds):
    """
    args
        masks: masks for all object boxes: (#boxes * mask_dim * mask_dim)
        pair_inds: pair indexes of potentional relationship
    return
        union_masks: ndarray of union_masks for all potentional relationship
    """
    union_masks = np.zeros((pair_inds.shape[0], masks.shape[-1], masks.shape[-1]))

    for i, pair in enumerate(pair_inds):
        union_masks[i] = np.maximum(masks[pair[0]], masks[pair[1]])
    return union_masks


def obj_union_position_coord(boxes, pair_inds):
    """
    args
        boxes: #boxes * [xmin, ymin, xmax, ymax]
        pair_inds: pair indexes of potentional relationship
    return
        union_coord: ndarray of union_coord for all potentional relationship
    """
    union_coord = np.zeros((pair_inds.shape[0], 4))

    for i, pair in enumerate(pair_inds):
        union_coord[i] = boxes[pair[0]] - boxes[pair[1]]
    return union_coord


def obj_union_semantic_feature(labels, pair_inds):
    """
    args
        labels: encoded labels for all objects
        pair_inds: pair indexes of potentional relationship
    return
        union_labels: ndarray of union_labels for all potentional relationship
    """
    union_labels = labels[pair_inds]
    return union_labels


def vg_collate(data, is_train=False, num_gpus=4, batch_size=1):
    blob = Blob(is_train=is_train, num_gpus=num_gpus, batch_size_per_gpu=len(data) // num_gpus)

    for d in data:
        if d != None:
            blob.append(d)
    blob.reduce()
    return blob


def assertion_checks(entry):
    num_gt = entry['gt_boxes'].shape[0]
    if entry['gt_classes'].shape[0] != num_gt:
        raise ValueError("GT classes and GT boxes must have same number of examples")

    assert (entry['gt_boxes'][:, 2] >= entry['gt_boxes'][:, 0]).all()
    assert (entry['gt_boxes'] >= -1).all()


def make_cluster_indices(n_boxes, max_batch_size, mode, minieval=False, validate_batch_size=False):
    validate_batch_size = False
    chunks_list = []

    bins = np.arange(5, 70, 5)

    if mode == 'val' or mode == 'test':
        num_batch = [100, 50, 20, 10, 5, 5, 3, 3, 1, 1, 1, 1, 1]  # 'nu2_eu3_lr00001_sg_ns'
        num_batch = [int(x * max_batch_size / 100) for x in num_batch]  # num_batch >= 1
    else:
        num_batch = [100, 50, 20, 10, 5, 5, 3, 3, 1, 1, 1, 1, 1]
        num_batch = [int(x * max_batch_size / 100) for x in num_batch]  # num_batch >= 1

    num_batch = [1 if x == 0 else x for x in num_batch]
    cluster = np.digitize(n_boxes, bins, right=True)

    for i in range(len(bins)):
        idxs = np.where(cluster == i)[0]

        if validate_batch_size == True:
            chunks = [idxs[x:x + num_batch[i]] for x in range(0, 1, num_batch[i])]
        elif (mode == 'val' or mode == 'test') and minieval == True:
            if 15 <= len(idxs):
                chunks = [idxs[x:x + num_batch[i]] for x in range(0, 15, num_batch[i])]
            else:
                chunks = [idxs[x:x + num_batch[i]] for x in range(0, len(idxs), num_batch[i])]
        else:
            chunks = [idxs[x:x + num_batch[i]] for x in range(0, len(idxs), num_batch[i])]

        if chunks == []:
            continue

        if len(chunks[0]) == 0:
            continue

        if i == 0:
            chunks_list = chunks
        else:
            chunks_list = chunks_list + chunks

    return chunks_list


def load_graphs_vrd(mode, dpath):
    gfile = '{tier}_sceneGraphs.pkl'.format(tier=mode) # _zeroshot for zeroshot test
    gpath = os.path.join(dpath, gfile)
    
    with open(gpath, 'rb') as f:
        data = pickle.load(f)

    img_idxs = []
    gt_classes = []
    gt_relationships = []
    gt_bboxes = []
    gt_nobjs = []
    gt_adjmats = []
    img_sizes = []

    key_list = list(data.keys())
    key_list.sort()

    for key in key_list:
        img_idxs.append(key)
        gt_nobjs.append(data[key]['num_objs'])
        gt_classes.append(np.array(data[key]['obj_labels']))
        gt_relationships.append(np.array(data[key]['edge_labels']))
        gt_bboxes.append(np.array(data[key]['obj_bboxes']))
        gt_adjmats.append(data[key]['adj'])

        img_size = (data[key]['height'], data[key]['width'], 1)
        img_sizes.append(img_size)

    return gt_classes, gt_relationships, gt_bboxes, gt_nobjs, gt_adjmats, img_sizes, img_idxs


def knowledge_prior(dpath, gt_classes, image_idxs, n_rel=71):
    prior = np.load(dpath + '/vrd_prior_traintest.npy')
    prior = prior / (np.sum(prior, axis=-1, keepdims=True) + 1e-10)

    kg_prior = []
    for i, idx in enumerate(image_idxs):
        classes = gt_classes[i]
        n_node = classes.shape[0]
        prior_mask = np.zeros((n_node, n_node, n_rel))

        for h_idx in range(n_node):
            for t_idx in range(n_node):
                if h_idx == t_idx:
                    continue
                else:
                    prior_mask[h_idx, t_idx] = prior[classes[h_idx], classes[t_idx]]

        kg_prior.append(prior_mask)
    return kg_prior

class ClusterRandomSampler(Sampler):
    """Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches (optional)

    Arguments:
    data_source (Dataset): a Dataset to sample from. Should have a cluster_indices property
    batch_size (int): a batch size that you would like to use later with Dataloader class
    shuffle (bool): whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_size, num_gpus, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.shuffle = shuffle
        self.batch_lists = self.make_batch_lists()

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def make_batch_lists(self):
        batch_lists = []

        for i, cluster_indices in enumerate(self.data_source.cluster_indices):

            batches = cluster_indices
            if self.shuffle:
                random.shuffle(batches)

            if len(batches) != self.batch_size:
                batch_pad = np.ones(self.batch_size)

                if self.num_gpus != 1 and len(batches) % self.num_gpus != 0:
                    continue
                else:
                    batch_pad[:] = np.nan
                    batch_pad[:len(batches)] = batches
                    batches = batch_pad

            batch_lists.append(batches)

        if self.shuffle:
            random.shuffle(batch_lists)

        batch_lists = self.flatten_list(batch_lists)
        return batch_lists

    def __iter__(self):
        return iter(self.batch_lists)

    def __len__(self):
        return len(self.batch_lists)


class DataLoader(data.DataLoader):
    @classmethod
    def get(cls, train_dset, val_dset, conf):
        train_loader = cls(
            train_dset,
            batch_size=conf.batch_size,
            num_workers=conf.nworker,
            collate_fn=lambda x: vg_collate(x, num_gpus=conf.ngpu, is_train=True, batch_size=conf.batch_size),
            shuffle=True,
            pin_memory=True,
            drop_last=True)

        val_loader = cls(
            val_dset,
            batch_size=conf.batch_size,
            num_workers=conf.nworker,
            collate_fn=lambda x: vg_collate(x, num_gpus=conf.ngpu, is_train=False, batch_size=conf.batch_size),
            shuffle=False,
            pin_memory=True,
            drop_last=True)

        return train_loader, val_loader