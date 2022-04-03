import os
from argparse import ArgumentParser
import numpy as np
import h5py

MODES = ('sgdet', 'sgcls', 'predcls', 'preddet')

class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        self.dataset = None
        self.lr = None
        self.odir = None
        self.opt = None
        self.ngpu = None
        self.epoch = None
        self.nworker = None
        self.batch_size = None
        self.dropout = None
        self.mode = None
        self.bmode = None
        self.val_size = None
        self.mtr_mode = None
        self.seed = None
        self.saved_model = None
        self.emb_dim = 300
        self.is_gtadj = None
        self.fgbg_ratio = None
        self.n_subspace = None
        self.comp_mode = None
        self.loss_mode = None
        self.is_fullmodel = False
        self.norel_mask = False
        self.kg_mask = False
        self.multi_pred=False
        self.alpha = None
        
        parser = self.setup_parser()
        args = vars(parser.parse_args())
        args['batch_size'] = args['batch_size']*args['ngpu']
        
        print("init: ## [config] ##")
        for x, y in args.items():
            print("{} : {}".format(x, y))
        print("##############")
        self.__dict__.update(args)

        if self.dataset == 'vg':
            self.DATA_PATH = '/root/storage/VG/'
            self.OUT_PATH = './results/'
            self.SF_PATH = self.DATA_PATH+'stanford_filtered/'
            self.IMG_PATH = self.DATA_PATH+'VG_100K'
            self.IMG_IDX = self.SF_PATH+'image_data.json'
            self.BOX_SCALE = 1024
            self.IM_SCALE = 592

        elif self.dataset == 'vrd':
            self.DATA_PATH = '/root/storage/VRD'
            self.OUT_PATH = './results/'
            self.IMG_PATH = None
            self.IMG_IDX = None

        elif self.dataset == 'vrr-vg':
            self.DATA_PATH = '/root/storage/VrR-VG'
            self.OUT_PATH = './results/'
            self.IMG_PATH = None
            self.IMG_IDX = None

        self.cpath = self.OUT_PATH + os.path.join(self.odir + '/config.json')

        if not os.path.exists(self.OUT_PATH):
            os.mkdir(self.OUT_PATH)

        if self.mode not in ('train', 'val', 'test'):
            raise ValueError('invalid mode: mode must be in {}'.format(('train', 'val', 'test')))

        if self.mtr_mode not in MODES:
            raise ValueError("Invalid mode: metric mode must be in {}".format(MODES))
            
        assert self.val_size >= 0

        if self.ngpu > 1:
            print("Let's use", self.ngpu, "GPUs!")
            
    def __call__(self):
        for x, y in self.__dict__.items():
            print("{} : {}".format(x, y))

    def setup_parser(self):
        """
        Sets up an argument parser
        :return: parser (ArgumentParser)
        """
        parser = ArgumentParser(description='model hyperparameters')

        parser.add_argument('-dataset', dest='dataset', help='dataset in {vg, gqa, vrd, vrr_vg}', type=str, default='vg')
        parser.add_argument('-ngpu', dest='ngpu', help='num gpus for training', type=int, default=1)
        parser.add_argument('-nwork', dest='nworker', help='num processes to use as workers', type=int, default=4)
        parser.add_argument('-batch_size', dest='batch_size', help='batch size per GPU', type=int, default=1000)
        parser.add_argument('-epoch', dest='epoch', help='max epoch', type=int, default=100)
        parser.add_argument('-lr', dest='lr', help='initial learning rate', type=float, default=0.001)
        parser.add_argument('-seed', dest='seed', help='seed number', type=int, default=1111)
        parser.add_argument('-bbox_mode', dest='bmode', help='bbox mode on (gt, ... )', default='gt')
        parser.add_argument('-val_size', dest='val_size', help='val size to use (if 0 we wont use val)', type=int, default=0)
        parser.add_argument('-mode', dest='mode', help='running mode in {train, val, test}', default='train')
        parser.add_argument('-loss_mode', dest='loss_mode', help='loss mode in {bce, ce}', default='bce')
        parser.add_argument('-opt', dest='opt', help='optimizer to train model', default='adam')
        parser.add_argument('-odir', dest='odir', help='directory to save trained model', default='test')
        parser.add_argument('-dropout', dest='dropout', help='dropout rate for encoder', type=float, default=0.0)
        parser.add_argument('-fgbg_ratio', dest='fgbg_ratio', help='ratio of background examples campared to foreground examples', type=float, default=0.3)
        parser.add_argument('-n_subspace', dest='n_subspace', help='number of latent relational subspace', type=int, default=5)
        parser.add_argument('-p_type', dest='p_type', help='positional feature type on (coord, mask)', default='coord')
        parser.add_argument('-is_gtadj', dest='is_gtadj', help='flag if the model use gtadj', action='store_true')
        parser.add_argument('-mtr_mode', dest='mtr_mode', help='metric mode in {sgdet, sgcls, predcls, preddet}', type=str, default='predcls')
        parser.add_argument('-comp_mode', dest='comp_mode', help='completion mode in {aver, waver, top1, topk}', type=str, default='aver')
        parser.add_argument('-norel_mask', dest='norel_mask', help='flag if the model use no-rel mask', action='store_true')
        parser.add_argument('-kg_mask', dest='kg_mask', help='flag if the model use knowledge prior mask', action='store_true')
        parser.add_argument('-saved_model', dest='saved_model', help='saved model name to retrain or evaulate', default=None)
        parser.add_argument('-alpha', dest='alpha', help='alpha value for loss', type=int, default=2)
        return parser