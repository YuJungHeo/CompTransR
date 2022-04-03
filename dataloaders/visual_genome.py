import os
import torch
import json
import math
import h5py
import numpy as np
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
        
        gpath = self.conf.SF_PATH + 'VG-SGG.h5'
        self.split_mask, self.gt_boxes, self.gt_nboxes, self.gt_classes, self.gt_relationships, self.idxs = load_graphs(gpath, self.mode, num_val_im=self.conf.val_size)
        self.kg_priors = knowledge_prior(self.conf.DATA_PATH, self.gt_classes, self.idxs)
        
        gdpath = self.conf.SF_PATH + 'VG-SGG-dicts.json'

        self.img_idxs = load_image_idxs(self.conf.IMG_IDX, self.conf.IMG_PATH)
        self.img_idxs = [self.img_idxs[i] for i in np.where(self.split_mask)[0]]            

        self.cluster_indices = make_cluster_indices(self.gt_nboxes, self.conf.batch_size, self.mode)
        self.num_batch = len(self.cluster_indices)
            
    def __len__(self):
        return (self.num_batch*self.conf.batch_size)

    def __getitem__(self, idx):
        if np.isnan(idx):
            return
        else:
            idx = int(idx)
        
        img_idx = self.img_idxs[idx]
        vnode = np.load(os.path.join(self.conf.DATA_PATH + 'VGSGG%s_node_%s/%s.npy' % (self.conf.bmode, self.mode, img_idx)))
        vedge = np.load(os.path.join(self.conf.DATA_PATH + 'VGSGG%s_edge_%s/%s.npy' % (self.conf.bmode, self.mode, img_idx)))
        
        gt_boxes = self.gt_boxes[idx].copy()
        scale = self.conf.IM_SCALE / self.conf.BOX_SCALE
        nnode = gt_boxes.shape[0]
        gt_boxes = gt_boxes.astype(np.float32) * scale
        gt_rels = self.gt_relationships[idx].copy()
        gt_classes = self.gt_classes[idx].copy()
        kg_priors = self.kg_priors[idx].copy()
        
        fn = os.path.join(self.conf.IMG_PATH, '%s.jpg'%(str(img_idx)))
        image_unpadded = Image.open(fn).convert('RGB')
        w, h = image_unpadded.size
        box_scale_factor = self.conf.BOX_SCALE / max(w, h)
        
        img_scale_factor = self.conf.IM_SCALE / max(w, h)
        if h > w:
            im_size = (self.conf.IM_SCALE, int(w * img_scale_factor), img_scale_factor)
        elif h < w:
            im_size = (int(h * img_scale_factor), self.conf.IM_SCALE, img_scale_factor)
        else:
            im_size = (self.conf.IM_SCALE, self.conf.IM_SCALE, img_scale_factor)

        all_pairs = np.transpose(np.mask_indices(nnode, np.triu, 1))
        
        if self.conf.p_type == 'coord':
            pnode = gt_boxes
            pedge = obj_union_position_coord(pnode, all_pairs)
            
        elif self.conf.p_type == 'mask':
            pnode = obj_position_mask(im_size, gt_boxes)
            pedge = obj_union_position_mask(pnode, all_pairs)

        if self.conf.mtr_mode == 'predcls':
            snode = gt_classes
            sedge = obj_union_semantic_feature(snode, all_pairs)
        
        gt_adjmat = adjmat(gt_classes.shape[0], gt_rels)
        
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

        masks[i, y1 : y2, x1 : x2] = 1
        assert(masks[i].sum() == (y2 - y1) * (x2 - x1))
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
    
def adjmat(n_obj, gt_rels):
    """
    args
        n_obj: number of objs in given image
        gt_rels: ground truth relationships
    return
        adjmat: adjcency matrix
    """
    adjmat = np.zeros((n_obj, n_obj))
    for rel in gt_rels:
        adjmat[rel[0], rel[1]] = rel[2]
    return adjmat
    
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
    
    bins = np.arange(5,70,5)
    # [5, 10, 15, 20, ... , 70] -> [B, n, n, d]
    if mode == 'val' or mode == 'test':
        num_batch = [100, 50, 20, 10, 5, 5, 3, 3, 1, 1, 1, 1, 1] # 'nu2_eu3_lr00001_sg_ns'
        num_batch = [int(x * max_batch_size / 100) for x in num_batch] # num_batch >= 1        
    else:    
        num_batch = [100, 50, 20, 10, 5, 5, 3, 3, 1, 1, 1, 1, 1]
        num_batch = [int(x * max_batch_size / 100) for x in num_batch] # num_batch >= 1
        
    num_batch = [1 if x == 0 else x for x in num_batch]
    cluster = np.digitize(n_boxes, bins, right=True)
    
    for i in range(len(bins)):
        idxs = np.where(cluster==i)[0]

        if validate_batch_size == True:
            chunks = [idxs[x:x+num_batch[i]] for x in range(0, 1, num_batch[i])]    
        elif (mode == 'val' or mode == 'test') and minieval == True:
            if 15 <= len(idxs):
                chunks = [idxs[x:x+num_batch[i]] for x in range(0, 15, num_batch[i])]
            else:
                chunks = [idxs[x:x+num_batch[i]] for x in range(0, len(idxs), num_batch[i])]
        else:
            chunks = [idxs[x:x+num_batch[i]] for x in range(0, len(idxs), num_batch[i])]
        
        if chunks == []:
            continue
            
        if len(chunks[0]) == 0:
            continue
            
        if i == 0:
            chunks_list = chunks
        else:
            chunks_list = chunks_list + chunks
    
    return chunks_list
    
def load_image_idxs(image_file, image_dir):
    """
    Loads the image ids from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    :param image_file: JSON file. Elements contain the param "image_id".
    :param image_dir: directory where the VisualGenome images are located
    :return: List of ids corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = [1592, 1722, 4616, 4617]
    img_id_list = []
    for i, img in enumerate(im_data):
        img_id = img['image_id']
        if img_id in corrupted_ims:
            continue
        img_id_list.append(img_id)
    assert len(img_id_list) == 108073
    return img_id_list

def load_graphs(gpath, mode, num_im=-1, num_val_im=0, filter_empty_rels=True, filter_non_overlap=False):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    :param graphs_file: HDF5
    :param mode: (train, val, or test)
    :param num_im: Number of images we want
    :param num_val_im: Number of validation images
    :param filter_empty_rels: (will be filtered otherwise.)
    :param filter_non_overlap: If training, filter images that dont overlap.
    :return: image_index: numpy array corresponding to the index of images we're using
                 boxes: List where each element is a [num_gt, 4] array of ground truth boxes (x1, y1, x2, y2)
                 gt_classes: List where each element is a [num_gt] array of classes
                 relationships: List where each element is a [num_r, 3] array of (box_ind_1, box_ind_2, predicate) relationships
    """

    graphs_file = gpath
    mode = mode
    BOX_SCALE = 1024
    
    roi_h5 = h5py.File(graphs_file, 'r')
    data_split = roi_h5['split'][:]
    split = 2 if mode == 'test' else 0
    split_mask = data_split == split

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if mode == 'val':
            image_index = image_index[:num_val_im]
        elif mode == 'train':
            image_index = image_index[num_val_im:]

    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    all_labels = roi_h5['labels'][:, 0]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]
    assert np.all(all_boxes[:, :2] >= 0)
    assert np.all(all_boxes[:, 2:] > 0)

    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    boxes = []
    num_boxes = []
    gt_classes = []
    relationships = []
    for i in range(len(image_index)):
        boxes_i = all_boxes[im_to_first_box[i]:im_to_last_box[i] + 1, :]
        gt_classes_i = all_labels[im_to_first_box[i]:im_to_last_box[i] + 1]

        if im_to_first_rel[i] >= 0:
            predicates = _relation_predicates[im_to_first_rel[i]:im_to_last_rel[i] + 1]
            obj_idx = _relations[im_to_first_rel[i]:im_to_last_rel[i] + 1] - im_to_first_box[i]
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert mode == 'train'
            inters = bbox_overlaps(boxes_i, boxes_i)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        rels = np.unique(rels, axis=0)
        boxes.append(boxes_i)
        num_boxes.append(boxes_i.shape[0])
        gt_classes.append(gt_classes_i)
        relationships.append(rels)
    return split_mask, boxes, num_boxes, gt_classes, relationships, image_index

def knowledge_prior(path, gt_classes, image_idxs, n_rel=51):
    prior = np.load(path + 'vg_prior.npy')
    
    c = 0
    prior = prior + c
    prior[prior>c] = 1
    
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
                batch_pad =  np.ones(self.batch_size)
                
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
        train_sampler = ClusterRandomSampler(train_dset, conf.batch_size, conf.ngpu, True)
        val_sampler = ClusterRandomSampler(val_dset, conf.batch_size, conf.ngpu, False)
        
        train_loader = cls(
                train_dset,
                batch_size=conf.batch_size,        
                num_workers=conf.nworker,
                collate_fn=lambda x: vg_collate(x, num_gpus=conf.ngpu, is_train=True, batch_size=conf.batch_size),
                sampler=train_sampler,
                shuffle=False,
                pin_memory=True,
                drop_last=False)
        
        val_loader = cls(
            val_dset, 
            batch_size=conf.batch_size, 
            num_workers=conf.nworker, 
            collate_fn=lambda x: vg_collate(x, num_gpus=conf.ngpu, is_train=False, batch_size=conf.batch_size), 
            sampler=val_sampler,
            shuffle=False, 
            pin_memory=True, 
            drop_last=False)    
        return train_loader, val_loader
