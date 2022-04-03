import json
import time
import resource
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import evaluate
from config import ModelConfig
from model.RelationLearner import RelLearner
from units.logger import Logger
from units.pytorch_misc import random_choose
from units.sg_eval import BasicSceneGraphEvaluator
from utils import *

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    
def init_setting(conf):
    """set optimizer and learning rate
    param:
        conf: configuration
    return:
        optimizer and scheduler
    """
    lr_default = conf.lr
    
    opt_name = conf.opt
    optimizer = optim.Adam(model.parameters(), lr=lr_default)
        
    lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.5, step=[60, 80, 120], repeat=1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    logger.print_model(model)
    logger.write(str(conf.__dict__))
    logger.write('optim: %s, lr=%.4f' % (opt_name, lr_default))
    return optimizer, scheduler

def MultiStepLR_Restart_Multiplier(epoch, gamma=0.1, step=[10,15,20], repeat=3):
    '''return the multipier for LambdaLR,
    0  <= ep < 10: gamma^0
    10 <= ep < 15: gamma^1
    15 <= ep < 20: gamma^2
    20 <= ep < 30: gamma^0 ... repeat 3 cycles and then keep gamma^2'''
    max_step = max(step)
    effective_epoch = epoch % max_step
    if epoch // max_step >= repeat:
        exp = len(step) - 1
    else:
        exp = len([i for i in step if effective_epoch>=i])
    return gamma ** exp

def train_epoch(epoch, train, val):
    model.train()
    edge_loss = 0
    total_loss = 0

    t = time.time()
    logger.write('lr: %f' % optimizer.param_groups[0]['lr'])
    writer.add_scalar('train_loss/lr', optimizer.param_groups[0]['lr'], epoch)
    train_loader, val_loader = DataLoader.get(train, val, conf)
    
    loss_for_sc = 0
    for b, batch in enumerate(tqdm(train_loader)):
        gt_edge = batch.gt_adjmats.long().cuda()
        
        res, pred_edge, gt_edge_selected, pred_edge_selected = train_batch(batch)
        loss_for_sc += res['total']

        if b % 100 == 0:
            n_topk = 5
            top1_edge = pred_edge.argmax(dim=1)
            _, top5_edge = pred_edge.topk(n_topk, dim=1)
            n_edge = top1_edge.shape[0]*top1_edge.shape[1]*top1_edge.shape[2]

            accu_edge_top1 = (top1_edge == gt_edge).float().sum()
            accu_edge_top5 = 0
            for i in range(n_topk):
                accu_edge_top5 += (top5_edge[:,i,:] == gt_edge).float().sum()
            assert accu_edge_top1 <= accu_edge_top5
            
            top1_pred_edge_selected = pred_edge_selected.argmax(dim=1)
            if conf.loss_mode == 'bce' or conf.loss_mode == 'margin':
                top1_gt_edge_selected = gt_edge_selected.argmax(dim=1)
            elif conf.loss_mode == 'ce':
                top1_gt_edge_selected = gt_edge_selected
            accu_edge_selected = (top1_gt_edge_selected == top1_pred_edge_selected).float().sum()
            n_selected = pred_edge_selected.shape[0]
            
            edge_loss = res['edge_loss']/n_edge
            prior_loss = res['prior_loss']/n_edge

            logger.write('train: %d th batch processed in %f, loss: %.5f, accu for edge(top1/top5/top1 selected): %.5f/%.5f/%.5f'
                         %(b, time.time()-t, edge_loss, accu_edge_top1/n_edge, accu_edge_top5/n_edge, accu_edge_selected/n_selected))
            
            writer.add_scalar('train_loss/edge_loss', edge_loss, epoch*len(train_loader)+b)
            writer.add_scalar('train_loss/prior_loss', prior_loss, epoch * len(train_loader) + b)
            writer.add_scalar('train_accu/accu_edge_top1', accu_edge_top1/n_edge, epoch*len(train_loader)+b)
            writer.add_scalar('train_accu/accu_edge_top5', accu_edge_top5/n_edge, epoch*len(train_loader)+b)
            writer.add_scalar('train_accu/accu_edge_selected', accu_edge_selected/n_selected, epoch*len(train_loader)+b)

    logger.write('train: %d th batch processed in %f, loss: %.5f, accu for edge(top1/top5/top1 selected): %.5f/%.5f/%.5f'
        % (b, time.time() - t, edge_loss, accu_edge_top1 / n_edge, accu_edge_top5 / n_edge, accu_edge_selected / n_selected))
    scheduler.step(epoch)

def train_batch(batch):
    gt_edge = batch.gt_adjmats.long().cuda()
    gt_adjmat = (batch.gt_adjmats > 0).float().cuda()
    kg_priors = batch.kg_priors.cuda()

    pred_edge, ht2rels, rel2rels = model[batch]
    n_subspace = pred_edge.shape[-1]

    gt_rels = batch.gt_rels
    gt_rels_sum = torch.sum(gt_rels, dim=-1)
    gt_rels_nonzero = gt_rels_sum.nonzero()
    fg_rels = gt_rels[gt_rels_nonzero[:, 0], gt_rels_nonzero[:, 1]]

    fg_pairs = gt_edge.nonzero()
    bg_all_pairs = (gt_edge == 0).nonzero()
    num_fg = fg_pairs.shape[0]
    num_bg = int (num_fg * conf.fgbg_ratio)
    bg_pairs = random_choose(bg_all_pairs, num_bg)

    bg_rels = torch.zeros(bg_pairs.shape[0], bg_pairs.shape[1]+1).long().cuda()
    bg_rels[:,:3] = bg_pairs

    fgbg_pairs = torch.cat((fg_pairs, bg_pairs), 0)
    _, perm = torch.sort(fgbg_pairs[:, 0]*(pred_edge.size(2)**2) + fgbg_pairs[:,1]*pred_edge.size(2) + fgbg_pairs[:,2])
    fgbg_pairs = fgbg_pairs[perm].contiguous()

    fgbg_rels = torch.cat((fg_rels, bg_rels), 0)
    _, perm = torch.sort(fgbg_rels[:, 0] * (pred_edge.size(2) ** 2) + fgbg_rels[:, 1] * pred_edge.size(2) + fgbg_rels[:, 2])
    fgbg_rels = fgbg_rels[perm].contiguous()

    if conf.mtr_mode == 'preddet':
        alpha = gt_adjmat

    vs_alpha = torch.ones_like(alpha) - alpha

    if conf.loss_mode == 'ce':
        final_logits = alpha[:, None, :, :, None] * pred_edge
    elif conf.loss_mode == 'bce' or conf.loss_mode == 'margin':
        final_logits = pred_edge
    final_logits[:, 0, :, :] = vs_alpha[:, :, :, None]
    pred_edge = final_logits

    if conf.loss_mode == 'bce':
        gt_edge_bce = torch.zeros_like(pred_edge)
        for rel in list(fg_rels):
            gt_edge_bce[rel[0], rel[3], rel[1], rel[2], :] = 1

        gt_edge_selected = gt_edge_bce[fgbg_pairs[:, 0], :, fgbg_pairs[:, 1], fgbg_pairs[:, 2]]
        pred_edge_selected = pred_edge[fgbg_pairs[:, 0], :, fgbg_pairs[:, 1], fgbg_pairs[:, 2], :]
        kg_priors_selected = kg_priors[fgbg_pairs[:, 0], fgbg_pairs[:, 1], fgbg_pairs[:, 2]].unsqueeze(-1).repeat(1, 1,pred_edge_selected.size(-1))
        ht2rels_selected = ht2rels[fgbg_pairs[:, 0], fgbg_pairs[:, 1], fgbg_pairs[:, 2], :]
        rel2rels_selected = rel2rels[fgbg_pairs[:, 0], fgbg_pairs[:, 1], fgbg_pairs[:, 2], :]

    elif conf.loss_mode == 'ce':
        gt_edge_selected = fgbg_rels[:,-1]
        pred_edge_selected = pred_edge[fgbg_rels[:,0], :, fgbg_rels[:,1], fgbg_rels[:,2]]

    elif conf.loss_mode == 'margin':
        gt_edge_bce = torch.zeros_like(pred_edge)
        gt_edge_bce[:, 0, :, :, :] = 1
        for rel in list(fg_rels):
            gt_edge_bce[rel[0], rel[3], rel[1], rel[2], :] = 1
            gt_edge_bce[rel[0], 0, rel[1], rel[2], :] = 0

        gt_edge_selected = gt_edge_bce[fgbg_pairs[:, 0], :, fgbg_pairs[:, 1], fgbg_pairs[:, 2]]

        gt_margin = torch.ones(fgbg_pairs.shape[0], pred_edge.shape[1])*(-1)
        for i, gt_pair in enumerate(gt_edge_selected):
            nz = gt_pair.nonzero()
            for j in range(len(nz)):
                gt_margin[i, j] = gt_pair.nonzero()[j][0]
        gt_margin = gt_margin.type(torch.cuda.LongTensor)
        pred_edge_selected = pred_edge[fgbg_pairs[:, 0], :, fgbg_pairs[:, 1], fgbg_pairs[:, 2]]
        kg_priors_selected = kg_priors[fgbg_pairs[:, 0], fgbg_pairs[:, 1], fgbg_pairs[:, 2]].unsqueeze(-1).repeat(1, 1, pred_edge_selected.size(-1))
        scores_selected = scores[fgbg_pairs[:, 0], :, fgbg_pairs[:, 1], fgbg_pairs[:, 2], :]

    losses = {}
    losses['edge_loss'] = 0
    losses['prior_loss'] = 0

    if conf.loss_mode == 'bce':
        losses['edge_loss'] = F.binary_cross_entropy(pred_edge_selected, gt_edge_selected)
        losses['prior_loss'] = F.binary_cross_entropy(pred_edge_selected, kg_priors_selected)
        loss = losses['edge_loss'] + conf.alpha * losses['prior_loss']
    elif conf.loss_mode == 'ce':
        for i in range(n_subspace):
            losses['edge_loss'] += F.nll_loss(pred_edge_selected[:,:,i], gt_edge_selected)
        loss = losses['edge_loss']
    elif conf.loss_mode == 'margin':
        criterion = nn.MultiLabelMarginLoss().cuda()
        for i in range(n_subspace):
            losses['edge_loss'] += criterion((pred_edge_selected[:, :, i]), gt_margin)
            losses['score_loss'] += criterion((scores_selected[:, :, i]), gt_margin)
        losses['prior_loss'] = F.binary_cross_entropy(pred_edge_selected, kg_priors_selected)
        loss = losses['edge_loss'] + conf.alpha * losses['prior_loss'] + losses['score_loss']

    pred_edge = torch.sum(pred_edge, dim=-1)/n_subspace
    pred_edge_selected = torch.sum(pred_edge_selected, dim=-1)/n_subspace
    if conf.loss_mode == 'bce':
        gt_edge_selected = torch.sum(gt_edge_selected, dim=-1)/n_subspace

    losses['total'] = loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    res = pd.Series({x: y.data for x, y in losses.items()})
    torch.cuda.empty_cache()
    return res, pred_edge, gt_edge_selected, pred_edge_selected

if __name__ == "__main__":
    conf = ModelConfig()
    fix_seed(conf.seed)

    if conf.dataset == 'vg':
        from dataloaders.visual_genome import Dataset, DataLoader
    elif conf.dataset == 'vrd':
        from dataloaders.vrd import Dataset, DataLoader
    elif conf.dataset == 'vrr-vg':
        from dataloaders.vrr_vg import Dataset, DataLoader
    elif conf.dataset == 'gqa':
        from dataloaders.gqa import Dataset, DataLoader

    out = conf.OUT_PATH + conf.odir
    logger = Logger(os.path.join(out, '%s_log.txt'%(conf.odir)))
    writer = SummaryWriter('results/'+conf.odir)

    config_file = os.path.join(out, '%s_config.json'%(conf.odir))

    if conf.saved_model != None:
        with open(config_file, "r") as confFile:
            conf_args = json.load(confFile)
            conf_args['saved_model'] = conf.saved_model
            conf.__dict__.update(conf_args)
            print("\nre-training: ## [config] ##")
            conf()

    train = Dataset(conf, 'train')
    val = Dataset(conf, 'test')
    test = Dataset(conf, 'test')
    train_loader, val_loader = DataLoader.get(train, val, conf)

    model = RelLearner(conf).cuda()
    optimizer, scheduler = init_setting(conf)

    if conf.saved_model != None:
        checkpoint = torch.load(os.path.join(out, conf.saved_model))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        start_epoch = epoch+1
        print (conf.saved_model, ' is loaded!')
    else:
        start_epoch = 0

    with open(config_file, "w") as confFile:
        json.dump(vars(conf), confFile)
    
    for epoch in range(start_epoch, start_epoch+conf.epoch):
        print ('epoch ', epoch)
        train_epoch(epoch, train, val)
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, os.path.join(out, 'ckpt_%d.pth.tar'%(epoch)))

        evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=conf.multi_pred)
        evaluate.eval_epoch(evaluator, conf, model, val, val_loader, logger, writer, epoch)
        