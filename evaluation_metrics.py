import os
import sys
import argparse
import time
import numpy as np
import datetime
import sklearn.metrics
from model import GPNN
from instrument_dataset import SurgicalDataset18


# Torch
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable



os.environ["CUDA_VISIBLE_DEVICES"] = "1"

INSTRUMENT_CLASSES = (
    '', 'kidney', 'bipolar_forceps', 'fenestrated_bipolar', 'prograsp_forceps', 'large_needle_driver', 'vessel_sealer',
    'grasping_retractor', 'monopolar_curved_scissors', 'ultrasound_probe', 'suction', 'clip_applier', 'stapler')

ACTION_CLASSES = (
    'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation', 'Tool_Manipulation', 'Cutting', 'Cauterization'
    , 'Suction', 'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing')


class Args:
    resume = 'ckpt/parsing/'
    visualize = False
    vis_top_k = 1

    # Optimization Options
    batch_size = 1
    no_cuda = False
    epochs = 100
    start_epoch = 0
    link_weight = 100
    lr = 1e-5
    lr_decay = 0.6
    momentum = 0.9
    log_interval = 200
    prefetch = 0
    #others
    ckpt_dir = 'ckpt/model/'


def evaluation(det_indices, pred_node_labels, node_labels, y_true, y_score, test=False):
    np_pred_node_labels = pred_node_labels.data.cpu().numpy()
    np_pred_node_labels_exp = np.exp(np_pred_node_labels)
    np_pred_node_labels = np_pred_node_labels_exp/(np_pred_node_labels_exp+1)  # overflows when x approaches np.inf
    np_node_labels = node_labels.data.cpu().numpy()

    new_y_true = np.empty((2 * len(det_indices), action_class_num))
    new_y_score = np.empty((2 * len(det_indices), action_class_num))
    for y_i, (batch_i, i, j) in enumerate(det_indices):
        new_y_true[2*y_i, :] = np_node_labels[batch_i, i, :]
        new_y_true[2*y_i+1, :] = np_node_labels[batch_i, j, :]
        new_y_score[2*y_i, :] = np_pred_node_labels[batch_i, i, :]
        new_y_score[2*y_i+1, :] = np_pred_node_labels[batch_i, j, :]

    y_true = np.vstack((y_true, new_y_true))
    y_score = np.vstack((y_score, new_y_score))
    return y_true, y_score


def weighted_loss(output, target):
    weight_mask = torch.autograd.Variable(torch.ones(target.size()))
    if hasattr(args, 'cuda') and args.cuda:
        weight_mask = weight_mask.cuda()
    link_weight = args.link_weight if hasattr(args, 'link_weight') else 1.0
    weight_mask += target * link_weight
    return torch.nn.MultiLabelSoftMarginLoss(weight=weight_mask).cuda()(output, target)


def loss_fn(pred_adj_mat, adj_mat, pred_node_labels, node_labels, mse_loss, multi_label_loss, human_num=[], obj_num=[]):
    np_pred_adj_mat = pred_adj_mat.data.cpu().numpy()
    det_indices = list()
    batch_size = pred_adj_mat.size()[0]
    loss = 0
    for batch_i in range(batch_size):
        valid_node_num = human_num[batch_i] + obj_num[batch_i]
        np_pred_adj_mat_batch = np_pred_adj_mat[batch_i, :, :]

        if len(human_num) != 0:
            human_interval = human_num[batch_i]
            obj_interval = human_interval + obj_num[batch_i]
        max_score = np.max([np.max(np_pred_adj_mat_batch), 0.01])
        mean_score = np.mean(np_pred_adj_mat_batch)

        batch_det_indices = np.where(np_pred_adj_mat_batch > 0.5)
        for i, j in zip(batch_det_indices[0], batch_det_indices[1]):
            # check validity for H-O interaction instead of O-O interaction
            if len(human_num) != 0:
                if i < human_interval and j < obj_interval:
                    if j >= human_interval:
                        det_indices.append((batch_i, i, j))

        loss = loss + weighted_loss(pred_node_labels[batch_i, :valid_node_num].view(-1, action_class_num), node_labels[batch_i, :valid_node_num].view(-1, action_class_num))
    return det_indices, loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




if __name__ == '__main__':
    args = Args()
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    action_class_num = 13
    hoi_class_num = 13
    edge_feature_size = 200
    node_feature_size = 200

    np.random.seed(0)
    torch.manual_seed(0)
    start_time = time.time()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    dataset = SurgicalDataset18(seq_set=[2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15], is_train=True)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    dataset_valid = SurgicalDataset18(seq_set=[1, 5, 16], is_train=True)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True)

    message_size = int(edge_feature_size/2)*2
    model_args = {'model_path': args.resume, 'edge_feature_size': edge_feature_size, 'node_feature_size': node_feature_size,
                  'message_size': message_size, 'link_hidden_size': 512,
                  'link_hidden_layers': 2, 'link_relu': False, 'update_hidden_layers': 1, 'update_dropout': False,
                  'update_bias': True, 'propagate_layers': 3,
                  'hoi_classes': action_class_num, 'resize_feature_to_message_size': False}

    model = GPNN(model_args)
    mse_loss = torch.nn.MSELoss(size_average=True)
    multi_label_loss = torch.nn.MultiLabelSoftMarginLoss(size_average=True)

    if args.cuda:
        model = model.cuda()
        gpu_ids = range(1)
        model = torch.nn.parallel.DataParallel(model, device_ids=gpu_ids)
        mse_loss = mse_loss.cuda()
        multi_label_loss = multi_label_loss.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epoch_errors = list()
    avg_epoch_error = np.inf
    best_epoch_error = np.inf

    best_epoch = 0
    best_mAP = 0
    best_loss = 0

    model.load_state_dict(torch.load("ckpt/model/epoch_14.pth.tar"))
    batch_time = AverageMeter()
    losses = AverageMeter()

    y_true = np.empty((0, action_class_num))
    y_score = np.empty((0, action_class_num))

    # switch to evaluate mode
    model.eval()
    print('size of valid sample:', len(valid_loader))
    end = time.time()
    for i, (edge_features, node_features, adj_mat, node_labels, file_name, human_num, obj_num) in enumerate(
            valid_loader):

        edge_features = torch.from_numpy(np.asarray(edge_features, np.float32)).float()
        node_features = torch.from_numpy(np.asarray(node_features, np.float32)).float()

        adj_mat = torch.from_numpy(np.asarray(adj_mat, np.float32)).float()
        node_labels = torch.from_numpy(np.asarray(node_labels, np.float32)).float()

        edge_features, node_features = Variable(edge_features).cuda(), Variable(node_features).cuda()
        adj_mat, node_labels = Variable(adj_mat).cuda(), Variable(node_labels).cuda()

        pred_adj_mat, pred_node_labels = model(edge_features, node_features, adj_mat, node_labels, human_num, obj_num,
                                               args)
        det_indices, loss = loss_fn(pred_adj_mat, adj_mat, pred_node_labels, node_labels, mse_loss, multi_label_loss,
                                    human_num, obj_num)
        if len(det_indices) > 0:
            losses.update(loss.data, len(det_indices))
            y_true, y_score = evaluation(det_indices, pred_node_labels, node_labels, y_true, y_score, test=False)
            # print(y_true.shape)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    avg_prec = sklearn.metrics.average_precision_score(y_true, y_score, average=None)
    mAP = np.nansum(avg_prec) / (len(avg_prec) - 3)

    # auc_all = []
    # for i in range(len(y_true)):
    #     if len(np.unique(y_true[i])) != 1:
    #         auc_all.append(sklearn.metrics.roc_auc_score(y_true[i], y_score[i]))
    # auc_avg = np.mean(np.array(auc_all))
    avg_auc = sklearn.metrics.roc_auc_score(np.array(y_true).flatten(), np.array(y_score).flatten())
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(y_true.flatten(), y_score.flatten())
    avg_recall = np.mean(recall)

    print('mAP:%.4f, AUC:%.4f, recall:%.4f'%(mAP, avg_auc, avg_recall))
