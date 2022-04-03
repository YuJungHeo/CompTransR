import time
import json
import resource
import dill as pkl

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import ModelConfig
from model.RelationLearner import RelLearner
from units.logger import Logger
from units.sg_eval import BasicSceneGraphEvaluator
from utils import *

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

def eval_batch(batch, conf, model, dataset, mid_mtrs):
    gt_edge = batch.gt_adjmats.long().cuda() # [B, n, n]
    gt_adjmat = (batch.gt_adjmats > 0).float().cuda() # [B, n, n]
    gt_nnodes = batch.gt_nnodes
    
    pred_edge, ht2rels, rel2rels = model[batch]
    n_subspace = pred_edge.shape[-1]

    pred_edge = torch.sum(pred_edge, dim=-1)/ n_subspace # [b, 51, n, n, #subs] -> [b, 51, n, n]

    if conf.mtr_mode == 'preddet':
        alpha = gt_adjmat

    vs_alpha = torch.ones_like(alpha) - alpha  # [b, n, n]
    final_logits = alpha[:, None, :, :] * pred_edge  # [b, 1, n, n]*[b, 51, n, n]
    final_logits[:, 0, :, :] = vs_alpha
    pred_edge = final_logits

    ##### ---- to evaluate top1 and top5 accuracy as a mid-metric
    gt_rels = batch.gt_rels
    gt_rels_sum = torch.sum(gt_rels, dim=-1) # [#batch, #rels]
    gt_rels_nonzero = gt_rels_sum.nonzero() # [ex, 2]
    fg_rels = gt_rels[gt_rels_nonzero[:,0], gt_rels_nonzero[:,1]]
    
    # --------------------------------------------- top1 and top5 accuracy (mid-metric)
    n_topk = 5

    top1_edge = pred_edge.argmax(dim=1) # [B, N, N]
    _, top5_edge = pred_edge.topk(n_topk, dim=1) # [B, 5, N, N]
    mid_mtrs['n_edge'] = mid_mtrs['n_edge'] + top1_edge.shape[0]*top1_edge.shape[1]*top1_edge.shape[2]
    mid_mtrs['accu_edge_top1'] = mid_mtrs['accu_edge_top1'] + (top1_edge == gt_edge).float().sum()
    for i in range(n_topk):
        mid_mtrs['accu_edge_top5'] = mid_mtrs['accu_edge_top5'] + (top5_edge[:,i,:] == gt_edge).float().sum()

    gt_edge_selected = fg_rels[:, -1]
    pred_edge_selected = pred_edge[fg_rels[:,0], :, fg_rels[:,1], fg_rels[:,2]]
    top1_pred_edge_selected = pred_edge_selected.argmax(dim=1)
    mid_mtrs['accu_edge_top1_selected'] = mid_mtrs['accu_edge_top1_selected'] + (gt_edge_selected == top1_pred_edge_selected).float().sum()
    mid_mtrs['n_edge_selected'] = mid_mtrs['n_edge_selected'] + pred_edge_selected.shape[0]
    # -------------------------------------------------------------------------------

    norel_mask = (top1_edge == 0)
    pred_edge = mask_pred(conf, pred_edge, gt_nnodes, batch.kg_priors, norel_mask)

    res = {}
    res['img_idx'], res['label_idx'], res['cols'], res['rows'], res['conf_vals'] = select_maxthrs(pred_edge, 100) # [b, k]
    assert sum(sum(res['label_idx']<0)) == 0
    torch.torch.cuda.empty_cache()
    return res, mid_mtrs, gt_adjmat, gt_nnodes
        
def eval_epoch(evaluator, conf, model, dataset, dataloader, logger, writer, epoch):
    model.eval()
    
    mid_mtrs = {}
    mid_mtrs['n_node'] = 0
    mid_mtrs['n_edge'] = 0
    mid_mtrs['n_edge_selected'] = 0    
    mid_mtrs['accu_node_top1'] = 0
    mid_mtrs['accu_edge_top1'] = 0
    mid_mtrs['accu_node_top5'] = 0
    mid_mtrs['accu_edge_top5'] = 0
    mid_mtrs['accu_edge_top1_selected'] = 0
    
    t = time.time()
    all_pred_entries = []
    all_adjmats = []
    
    for b, batch in enumerate(tqdm(dataloader)):
        res, mid_mtrs, gt_adjmat, gt_nnodes = eval_batch(batch, conf, model, dataset, mid_mtrs)

        for i, idx in enumerate(batch.idx_list):
            gt_entry = {
                'gt_classes': dataset.gt_classes[idx].copy(),
                'gt_relations': dataset.gt_relationships[idx].copy(),
                'gt_boxes': dataset.gt_boxes[idx].copy(),
            }

            pred_entry = {
                'pred_boxes': None,
                'pred_classes': None,
                'pred_rel_inds': torch.stack((res['cols'][i], res['rows'][i])).transpose(0, 1).detach().cpu().numpy(),
                'obj_scores': None,
                'rel_inds': res['label_idx'][i].detach().cpu().numpy(),
                'rel_scores': res['conf_vals'][i].detach().cpu().numpy(),
                'idx': idx
            }
            all_pred_entries.append(pred_entry)
            evaluator[conf.mtr_mode].evaluate_scene_graph_entry(
                gt_entry,
                pred_entry,
            )

    eval_time = time.time() - t
    
    accu_edge_top1 = mid_mtrs['accu_edge_top1']/mid_mtrs['n_edge']
    accu_edge_top5 = mid_mtrs['accu_edge_top5']/mid_mtrs['n_edge']
    accu_edge_selected = mid_mtrs['accu_edge_top1_selected']/mid_mtrs['n_edge_selected']
    logger.write('eval: time: %f, accu for edge(top1/top5/top1 selected): %.5f/%.5f/%.5f' % (eval_time, accu_edge_top1, accu_edge_top5, accu_edge_selected))

    writer.add_scalar('val_accu/accu_edge_top1', accu_edge_top1, epoch)
    writer.add_scalar('val_accu/accu_edge_top5', accu_edge_top5, epoch)
    writer.add_scalar('val_accu/accu_edge_selected', accu_edge_selected, epoch)
    
    evaluator[conf.mtr_mode].print_stats_log(logger, writer, epoch)
        
    out_file = conf.OUT_PATH + conf.odir + '/' + conf.odir + '_e%s.pkl'%(epoch)
    with open(out_file,'wb') as f:
        pkl.dump(all_pred_entries, f)
    
def mask_pred(conf, pred, gt_nnodes, priors, norel_mask):
    """
    mask background relationships and values at padded pairs
    args
        pred: prediction results # [b, 51, n, n]
        gt_nnodes : number of gt_nodes in an image # [b, 1]
        norel_mask: [b, n, n]
    """

    if conf.kg_mask == True:
        priors = priors.permute(0, 3, 1, 2) # [b, 51, head, tail]
        pred = torch.mul(pred, priors)

    if conf.norel_mask == True:
        norel_mask = norel_mask.repeat(pred.shape[1], 1, 1, 1).permute(1, 0, 2 ,3)
        norel_mask = norel_mask.float()*1e-2
        norel_mask = norel_mask + (norel_mask == 0).float()
        pred = torch.mul(pred, norel_mask)


    pred = mask2d_4dmat(gt_nnodes, pred, val=-1)
    pred[:,0,:,:] = -1
    return pred

def select_maxthrs(pred, max_thrs):
    """
    select max_thrs pairs following high confidence (it helps to save results efficiently)
    args
        pred: prediction results # [b, 51, n, n]
        max_thrs: max threshold value
    """
    n_node = pred.shape[-1]
    pred_flat = pred.contiguous().view(pred.shape[0], -1)
    pred_score, max_idx = pred_flat.topk(k=max_thrs)
    label_idx = max_idx / (n_node*n_node)
    remains = (max_idx-n_node*n_node*label_idx)
    cols = remains / n_node
    rows = remains % n_node
    img_idx = torch.arange(0, pred.shape[0]).repeat(max_thrs, 1).transpose(0, 1)
    conf_vals = pred[img_idx, label_idx, cols, rows]
    return img_idx, label_idx, cols, rows, conf_vals

if __name__ == '__main__':
    conf = ModelConfig()
    assert conf.saved_model != None

    fix_seed(conf.seed)

    out = conf.OUT_PATH + conf.odir
    logger = Logger(os.path.join(out, '%s_log.txt' % (conf.odir)))
    writer = SummaryWriter('results/' + conf.odir)

    # evaluate: conf.saved_model != None
    config_file = os.path.join(out, '%s_config.json' % (conf.odir))
    with open(config_file, "r") as confFile:
        conf_args = json.load(confFile)
        conf_args['saved_model'] = conf.saved_model
        conf_args['mode'] = 'test'
        conf.__dict__.update(conf_args)
        print("\nevaluate: ## [config] ##")
        conf()

    if conf.dataset == 'vg':
        from dataloaders.visual_genome import Dataset, DataLoader
    elif conf.dataset == 'vrd':
        from dataloaders.vrd import Dataset, DataLoader
    elif conf.dataset == 'vrr_vg':
        from dataloaders.vrr_vg import Dataset, DataLoader
    elif conf.dataset == 'gqa':
        from dataloaders.gqa import Dataset, DataLoader

    if conf.mode == 'train' or conf.mode == 'val':
        eval_data = Dataset(conf, 'val')
        print ('Evaluating val dataset starts now!')
    elif conf.mode == 'test':
        eval_data = Dataset(conf, 'test')
        print ('Evaluating test dataset starts now!')
    _, eval_loader = DataLoader.get(eval_data, eval_data, conf)

    model = RelLearner(conf).cuda()
    opt_name = conf.opt
    if opt_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9)
    elif opt_name == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=conf.lr)
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)

    checkpoint = torch.load(os.path.join(out, conf.saved_model))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    
    evaluator = BasicSceneGraphEvaluator.all_modes()
    eval_epoch(evaluator, conf, model, eval_data, eval_loader, logger, writer, epoch)