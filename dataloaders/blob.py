"""
Data blob, hopefully to make collating less painful and MGPU training possible
"""
import numpy as np
import torch
from torch.autograd import Variable

class Blob(object):
    def __init__(self, is_train=False, num_gpus=1, primary_gpu=0, batch_size_per_gpu=0):
        """
        Initializes an empty Blob object.
        :param is_train: True if it's training
        """
        assert num_gpus >= 1
        
        self.is_train = is_train
        self.num_gpus = num_gpus
        self.batch_size_per_gpu = batch_size_per_gpu
        self.primary_gpu = primary_gpu

        self.idx_list = []
        self.im_sizes = []  # [num_images, 3] array of (h, w, scale)
        self.gt_boxes = []  # [num_gt, 4] boxes
        self.gt_classes = []  # [num_gt,2] array of img_ind, class
        self.gt_rels = []  # [num_rels, 3]. Each row is (gtbox0, gtbox1, rel).
        self.gt_adjmats = []
        self.gt_nnodes = []
        self.kg_priors = []
        
        self.vnode = []
        self.vedge = []
        self.pnode = []
        self.pedge = []
        self.snode = []
        self.sedge = []
        
    @property
    def volatile(self):
        return not self.is_train

    def append(self, d):
        """
        Adds a single image to the blob
        :param datom:
        :return:
        """
        
        i = len(self.im_sizes)
        h, w, scale = d['img_size']
        self.im_sizes.append((h, w, scale))

        self.idx_list.append(d['index'])
        self.gt_boxes.append(d['gt_boxes'])
        self.gt_classes.append(np.column_stack((i * np.ones(d['gt_classes'].shape[0], dtype=np.int64), d['gt_classes'])))
        self.gt_rels.append(np.column_stack((i * np.ones(d['gt_relations'].shape[0], dtype=np.int64), d['gt_relations'])))
        self.kg_priors.append(d['kg_priors'])
        self.vnode.append(d['vnode'])
        self.vedge.append(d['vedge'])
        self.pnode.append(d['pnode'])
        self.pedge.append(d['pedge'])
        self.snode.append(d['snode'])
        self.sedge.append(d['sedge'])
        self.gt_adjmats.append(d['gt_adjmat'])
        self.gt_nnodes.append(d['gt_nnodes'])
        
    def _chunkize(self, datom, tensor=torch.LongTensor, padding=True, nsquare=False):
        """
        Turn data list into chunks, one per GPU
        :param datom: List of lists of numpy arrays that will be concatenated.
        :return:
        """
        
        chunk_sizes = [0] * self.num_gpus
        if padding == True:   
            max_obj = max([d.shape[0] for d in datom])

        for i in range(self.num_gpus):
            for j in range(self.batch_size_per_gpu):
                if padding == True:
                    d = datom[i * self.batch_size_per_gpu + j]
                    
                    if nsquare == True: # only adjmat and kg_prior
                        if d.ndim == 2:
                            d_pad = np.zeros((1, max_obj, max_obj))
                            d_pad[0, :d.shape[0], :d.shape[0]] = d
                            datom[i * self.batch_size_per_gpu + j] = d_pad
                        elif d.ndim == 3:
                            d_pad = np.zeros((1, max_obj, max_obj, d.shape[-1]))
                            d_pad[0, :d.shape[0], :d.shape[0], :] = d
                            datom[i * self.batch_size_per_gpu + j] = d_pad
                            
                    elif len(d.shape) == 1:
                        d_pad = np.zeros((1, max_obj, ))
                        d_pad[0, :d.shape[0]] = d
                        datom[i * self.batch_size_per_gpu + j] = d_pad
                        
                    elif len(d.shape) == 2:
                        d_pad = np.zeros((1, max_obj, d.shape[-1]))
                        d_pad[0, :d.shape[0], :] = d
                        datom[i * self.batch_size_per_gpu + j] = d_pad
                        
                    elif len(d.shape) == 3:
                        d_pad = np.zeros((1, max_obj, d.shape[-1], d.shape[-1]))
                        d_pad[0, :d.shape[0], :, :] = d
                        datom[i * self.batch_size_per_gpu + j] = d_pad
                        
                chunk_sizes[i] += datom[i * self.batch_size_per_gpu + j].shape[0]
        return Variable(tensor(np.concatenate(datom, 0))), chunk_sizes

    def reduce(self):
        """ Merges all the detections into flat lists + numbers of how many are in each"""
        
        """ deprecated
        if len(self.im_sizes) != self.batch_size_per_gpu * self.num_gpus:
            raise ValueError("Wrong batch size? imgs len {} bsize/gpu {} numgpus {}".format(
                len(self.im_sizes), self.batch_size_per_gpu, self.num_gpus
            ))
        """
        num_data = len(self.im_sizes)
        self.batch_size_per_gpu = int(num_data/self.num_gpus)
        
        self.im_sizes = np.stack(self.im_sizes).reshape((self.num_gpus, self.batch_size_per_gpu, 3))

        self.gt_nnodes = Variable(torch.LongTensor(self.gt_nnodes))
        self.gt_rels, self.gt_rel_chunks = self._chunkize(self.gt_rels) # maybe not used
        self.gt_boxes, self.node_chunks = self._chunkize(self.gt_boxes, tensor=torch.FloatTensor) # maybe not used
        self.gt_classes, _ = self._chunkize(self.gt_classes)
        self.vnode, _ = self._chunkize(self.vnode, tensor=torch.FloatTensor)
        self.vedge, self.edge_chunks = self._chunkize(self.vedge, tensor=torch.FloatTensor)
        self.pnode, _ = self._chunkize(self.pnode, tensor=torch.FloatTensor)
        self.pedge, _ = self._chunkize(self.pedge, tensor=torch.FloatTensor)
        self.snode, _ = self._chunkize(self.snode, tensor=torch.FloatTensor)
        self.sedge, _ = self._chunkize(self.sedge, tensor=torch.FloatTensor)
        self.gt_adjmats, _ = self._chunkize(self.gt_adjmats, tensor=torch.FloatTensor, nsquare=True)
        self.kg_priors, _ = self._chunkize(self.kg_priors, tensor=torch.FloatTensor, nsquare=True)
        
    def _scatter(self, x, chunk_sizes, dim=0):
        """ Helper function"""
        if self.num_gpus == 1:
            #return x.cuda(self.primary_gpu, async=True)
            return x.cuda(self.primary_gpu)
        return torch.nn.parallel.scatter_gather.Scatter.apply(list(range(self.num_gpus)), chunk_sizes, dim, x)
    
    def scatter(self):
        """ Assigns everything to the GPUs"""
        #self.gt_classes_primary = self.gt_classes.cuda(self.primary_gpu, async=True)
        #self.gt_boxes_primary = self.gt_boxes.cuda(self.primary_gpu, async=True)
        #self.gt_nnodes = self.gt_nnodes.cuda(self.primary_gpu, async=True)
        ## TODO!!!: maybe have to process gt_nnodes
        self.gt_classes = self._scatter(self.gt_classes, self.node_chunks)
        self.gt_boxes = self._scatter(self.gt_boxes, self.node_chunks)
        self.vnode = self._scatter(self.vnode, self.node_chunks)
        self.vedge = self._scatter(self.vedge, self.edge_chunks)
        self.pnode = self._scatter(self.pnode, self.node_chunks)
        self.pedge = self._scatter(self.pedge, self.edge_chunks)
        self.snode = self._scatter(self.snode, self.node_chunks)
        self.sedge = self._scatter(self.sedge, self.edge_chunks)
        self.gt_rels = self._scatter(self.gt_rels, self.gt_rel_chunks)
        self.gt_adjmats = self._scatter(self.gt_adjmats, self.node_chunks)
        self.kg_priors = self._scatter(self.kg_priors, self.node_chunks)
        
        """
        if self.is_train:
            self.gt_rels = self._scatter(self.gt_rels, self.gt_rel_chunks)
            self.gt_adjmats = self._scatter(self.gt_adjmats, self.node_chunks)
            self.kg_priors = self._scatter(self.kg_priors, self.node_chunks)
        else:
            self.gt_rels = self.gt_rels.cuda(self.primary_gpu, async=True)
            self.gt_adjmats = self.gt_adjmats.cuda(self.primary_gpu, async=True)
            self.kg_priors = self.kg_priors.cuda(self.primary_gpu, async=True)
        """ 
    def __getitem__(self, index):
        """
        Returns a tuple containing data
        :param index: Which GPU we're on, or 0 if no GPUs
        :return: If training: (im_size, img_start_ind, gt_boxes, gt_classes, vnode, vedge, gt_rels) / if test: (im_size, img_start_ind, gt_boxes, gt_classes, vnode, vedge)
        """
        
        if index not in list(range(self.num_gpus)):
            raise ValueError("Out of bounds with index {} and {} gpus".format(index, self.num_gpus))

        if index == 0 and self.num_gpus == 1:
            image_offset = 0
            if self.is_train:
                return (self.im_sizes[0], image_offset, self.gt_boxes, self.gt_classes, self.vnode, self.vedge, self.pnode, self.pedge, self.snode, self.sedge, self.gt_adjmats, self.gt_rels, self.gt_nnodes, self.kg_priors)
            return (self.im_sizes[0], image_offset, self.gt_boxes, self.gt_classes, self.vnode, self.vedge, self.pnode, self.pedge, self.snode, self.sedge, self.gt_adjmats, self.gt_rels, self.gt_nnodes, self.kg_priors)
        else: 
            image_offset = self.batch_size_per_gpu * index

            if self.is_train:
                return (self.im_sizes[index], image_offset, self.gt_boxes[index], self.gt_classes[index], 
                        self.vnode[index], self.vedge[index], self.pnode[index], self.pedge[index], self.snode[index], self.sedge[index], self.gt_adjmats[index], self.gt_rels[index])
            return (self.im_sizes[index], image_offset, self.gt_boxes[index], self.gt_classes[index], 
                        self.vnode[index], self.vedge[index], self.pnode[index], self.pedge[index], self.snode[index], self.sedge[index], self.gt_adjmats[index], self.gt_rels[index])