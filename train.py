from net.nodule_net import NoduleNet
import time
from dataset.collate import train_collate, test_collate, eval_collate
from dataset.bbox_reader import BboxReader
from dataset.mask_reader import MaskReader
from utils.util import Logger
from config import train_config, data_config, net_config, config
import pprint
from torch.utils.data import DataLoader, ConcatDataset
#from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
import torch
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
import random
import traceback
from torch.utils.tensorboard import SummaryWriter
import torchvision
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from PIL import Image, ImageDraw

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

this_module = sys.modules[__name__]
print("Is CUDA available: ", torch.cuda.is_available())

parser = argparse.ArgumentParser(description='PyTorch Detector')
parser.add_argument('--net', '-m', metavar='NET', default=train_config['net'],
                    help='neural net')
parser.add_argument('--epochs', default=train_config['epochs'], type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=train_config['batch_size'], type=int, metavar='N',
                    help='batch size')
parser.add_argument('--epoch-rcnn', default=train_config['epoch_rcnn'], type=int, metavar='NR',
                    help='number of epochs before training rcnn')
parser.add_argument('--epoch-lhi', default=train_config['epoch_lhi'], type=int, metavar='NR',
                    help='number of epochs before training lhi')
parser.add_argument('--epoch-mask', default=train_config['epoch_mask'], type=int, metavar='NR',
                    help='number of epochs before training mask branch')
parser.add_argument('--ckpt', default=train_config['initial_checkpoint'], type=str, metavar='CKPT',
                    help='checkpoint to use')
parser.add_argument('--optimizer', default=train_config['optimizer'], type=str, metavar='SPLIT',
                    help='which split set to use')
parser.add_argument('--init-lr', default=train_config['init_lr'], type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=train_config['momentum'], type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=train_config['weight_decay'], type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--epoch-save', default=train_config['epoch_save'], type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--out-dir', default=train_config['out_dir'], type=str, metavar='OUT',
                    help='directory to save results of this training')
parser.add_argument('--train-set-list', default=train_config['train_set_list'], nargs='+', type=str,
                    help='train set paths list')
parser.add_argument('--val-set-list', default=train_config['val_set_list'], nargs='+', type=str,
                    help='val set paths list')
parser.add_argument('--data-dir', default=train_config['DATA_DIR'], type=str, metavar='OUT',
                    help='path to load data')
parser.add_argument('--lungmask-dir', default=train_config['LUNGMASK_DIR'], type=str, metavar='OUT',
                    help='path to load lung mask')
parser.add_argument('--num-workers', default=train_config['num_workers'], type=int, metavar='N',
                    help='number of data loading workers')
#parser.add_argument('--distributed', default=net_config['distributed'], type=bool, help='if ddp is used')
#parser.add_argument('--local_rank', type=int, default=0)

def main():
    # Load training configuration
    args = parser.parse_args()
    #print(args.distributed)
    #if args.distributed:
    #    torch.distributed.init_process_group(backend="nccl", init_method='env://')

    net = args.net
    initial_checkpoint = args.ckpt
    out_dir = args.out_dir
    weight_decay = args.weight_decay
    momentum = args.momentum
    optimizer = args.optimizer
    init_lr = args.init_lr
    epochs = args.epochs
    epoch_save = args.epoch_save
    epoch_rcnn = args.epoch_rcnn
    epoch_lhi = args.epoch_lhi
    epoch_mask = args.epoch_mask
    batch_size = args.batch_size
    train_set_list = args.train_set_list
    val_set_list = args.val_set_list
    num_workers = args.num_workers
    lr_schdule = train_config['lr_schedule']
    data_dir = args.data_dir
    lungmask_dir = args.lungmask_dir
    label_types = config['label_types']

    train_dataset_list = []
    val_dataset_list = []
    for i in range(len(train_set_list)):
        set_name = train_set_list[i]
        label_type = label_types[i]

        if label_type == 'bbox':
            dataset = BboxReader(data_dir, set_name, config, mode='train', lungmask_dir=lungmask_dir)
        elif label_type == 'mask':
            dataset = MaskReader(data_dir, set_name, config, mode='train')

        train_dataset_list.append(dataset)

    for i in range(len(val_set_list)):
        set_name = val_set_list[i]
        label_type = label_types[i]

        if label_type == 'bbox':
            dataset = BboxReader(data_dir, set_name, config, mode='val', lungmask_dir=lungmask_dir)
        elif label_type == 'mask':
            dataset = MaskReader(data_dir, set_name, config, mode='val')
            
        val_dataset_list.append(dataset)
    
    train_dataset = ConcatDataset(train_dataset_list)
    #if args.distributed:
    #    train_sampler = DistributedSampler(train_dataset)
    #else:
    train_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                              num_workers=num_workers, pin_memory=True, collate_fn=train_collate, sampler=train_sampler)

    val_dataset = ConcatDataset(val_dataset_list)
    #if args.distributed:
    #    val_sampler = DistributedSampler(val_dataset)
    #else:
    val_sampler = None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, collate_fn=train_collate, sampler=val_sampler)
    
    
    # Initilize network
    net = getattr(this_module, net)(net_config)
    net = net.cuda()
    print("Are model params in cuda: ", next(net.parameters()).is_cuda) # True
    optimizer = getattr(torch.optim, optimizer)
    # optimizer = optimizer(net.parameters(), lr=init_lr, weight_decay=weight_decay)
    optimizer = optimizer(net.parameters(), lr=init_lr, weight_decay=weight_decay, momentum=momentum)

    start_epoch = 0

    if initial_checkpoint:
        print('[Loading model from %s]' % initial_checkpoint)
        checkpoint = torch.load(initial_checkpoint)
        start_epoch = checkpoint['epoch']
        state = net.state_dict()
        state.update(checkpoint['state_dict'])

        try:
            net.load_state_dict(state)
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print('Load something failed!')
            traceback.print_exc()

    start_epoch = start_epoch + 1

    model_out_dir = os.path.join(out_dir, 'model')
    tb_out_dir = os.path.join(out_dir, 'runs')
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    logfile = os.path.join(out_dir, 'log_train')
    sys.stdout = Logger(logfile)

    print('[Training configuration]')
    for arg in vars(args):
        print(arg, getattr(args, arg))

    print('[Model configuration]')
    pprint.pprint(net_config)

    print('[start_epoch %d, out_dir %s]' % (start_epoch, out_dir))
    print('[length of train loader %d, length of valid loader %d]' % (len(train_loader), len(val_loader)))

    # Write graph to tensorboard for visualization
    writer = SummaryWriter(tb_out_dir)
    train_writer = SummaryWriter(os.path.join(tb_out_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(tb_out_dir, 'val'))
    # writer.add_graph(net, (torch.zeros((16, 1, 128, 128, 128)).cuda(), [[]], [[]], [[]], [torch.zeros((16, 128, 128, 128))]), verbose=False)

    for i in tqdm(range(start_epoch, epochs + 1), desc='Total'):

        #if args.distributed:
        #    train_sampler.set_epoch(i)
        
        # learning rate schedule
        if isinstance(optimizer, torch.optim.SGD):
            lr = lr_schdule(i, init_lr=init_lr, total=epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = np.inf

        if i >= epoch_rcnn:
            net.use_rcnn = True
        else:
            net.use_rcnn = False
        if i >= epoch_lhi:
            net.use_lhi = True
        else:
            net.use_lhi = False
        if i >= epoch_mask:
            net.use_mask = True
        else:
            net.use_mask = False

        print('[epoch %d, lr %f, use_rcnn: %r, use_mask: %r, use_lhi: %r]' % (i, lr, net.use_rcnn, net.use_mask, net.use_lhi))
        train(net, train_loader, optimizer, i, train_writer)
        validate(net, val_loader, i, val_writer)
        print

        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        if i % epoch_save == 0:
            torch.save({
                'epoch': i,
                'out_dir': out_dir,
                'state_dict': state_dict,
                'optimizer' : optimizer.state_dict()},
                os.path.join(model_out_dir, '%03d.ckpt' % i))

    writer.close()
    train_writer.close()
    val_writer.close()

def plot_bbox_and_lhi(img, pred_bboxes, gt_bboxes, crop_lhi, crop_ims, width=1, resize_scale=2, thres=0.7):
    B, _, D, H, W = img.shape
    pred_bboxes = [[pred_bbox for pred_bbox in pred_bboxes if pred_bbox[0]==b] for b in range(B)]
    draw_img = []
    lhi_img = []
    crop_img = []
    idx_lhi = 0
    for b in range(B):
        idx = 0
        found = 0
        for i in range(len(pred_bboxes[b])):
            if pred_bboxes[b][i][1] >= thres:
                pred_bbox = pred_bboxes[b][i]
                found = 1
                break
            idx += 1

        if found:
            idx_lhi += idx
            lhi_img.append(crop_lhi[idx_lhi])
            if crop_ims is not None:
                unnorm_crop_img = crop_ims[idx_lhi][0,int(crop_ims[idx_lhi].shape[1]/2)]*128+128
                crop_ndarr = unnorm_crop_img.div(unnorm_crop_img.max()).mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
                crop_im = Image.fromarray(crop_ndarr).convert('RGB').resize((48,48))
                crop_img.append(torch.from_numpy(np.array(crop_im)).permute(2,1,0))
            idx_lhi += len(pred_bboxes[b][idx:])

        else:
            idx_lhi += len(pred_bboxes[b])
            continue

        _, _, z_start, y_start, x_start, z_end, y_end, x_end = pred_bbox
        z = int((z_start + z_end)/2)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        unnorm_img = img[b,0,z]*128+128
        ndarr = unnorm_img.div(unnorm_img.max()).mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr).convert('RGB').resize((W*resize_scale, H*resize_scale))
        draw = ImageDraw.Draw(im)
        draw.rectangle([int(x_start*resize_scale), int(y_start*resize_scale), int(x_end*resize_scale), int(y_end*resize_scale)], outline=(255,0,0), width=width)
        
        for next_pred_bbox in pred_bboxes[b][idx_lhi+1:]:
            _, p, next_z_start, next_y_start, next_x_start, next_z_end, next_y_end, next_x_end = next_pred_bbox
            next_z = int((next_z_start + next_z_end)/2)
            if p >= thres and z_start <= next_z <= z_end:
                draw.rectangle([int(next_x_start*resize_scale), int(next_y_start*resize_scale), int(next_x_end*resize_scale), int(next_y_end*resize_scale)], outline=(0,0,255), width=width)
        
        for gt_bbox in gt_bboxes[b]:
            gt_z, gt_y, gt_x, gt_d, gt_h, gt_w = gt_bbox
            if z_start <= gt_z <= z_end:
                gt_x_start, gt_y_start, gt_x_end, gt_y_end =\
                int(gt_x - gt_w/2), int(gt_y - gt_h/2), int(gt_x + gt_w/2), int(gt_y + gt_h/2)
                draw.rectangle([int(gt_x_start*resize_scale), int(gt_y_start*resize_scale), int(gt_x_end*resize_scale), int(gt_y_end*resize_scale)], outline=(0,255,0), width=width)

        draw_img.append(torch.from_numpy(np.array(im)).permute(2,1,0))
    return draw_img, lhi_img

def train(net, train_loader, optimizer, epoch, writer):
    net.set_mode('train')
    s = time.time()
    rpn_cls_loss, rpn_reg_loss = [], []
    rcnn_cls_loss, rcnn_reg_loss = [], []
    lhi_cls_loss, lhi_reg_loss = [], []
    mask_loss = []
    total_loss = []
    rpn_stats = []
    rcnn_stats = []
    lhi_stats = []
    mask_stats = []

    for j, (input, truth_box, truth_label, truth_mask, masks) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train %d' % epoch):
        input = Variable(input).cuda()
        truth_box = np.array(truth_box)
        truth_label = np.array(truth_label)
        truth_mask = np.array(truth_mask)

        net(input, truth_box, truth_label, truth_mask, masks)

        loss, rpn_stat, rcnn_stat, mask_stat, lhi_stat = net.loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rpn_cls_loss.append(net.rpn_cls_loss.cpu().data.item())
        rpn_reg_loss.append(net.rpn_reg_loss.cpu().data.item())
        rcnn_cls_loss.append(net.rcnn_cls_loss.cpu().data.item())
        rcnn_reg_loss.append(net.rcnn_reg_loss.cpu().data.item())
        lhi_cls_loss.append(net.lhi_cls_loss.cpu().data.item())
        lhi_reg_loss.append(net.lhi_reg_loss.cpu().data.item())
        mask_loss.append(net.mask_loss.cpu().data.item())
        total_loss.append(loss.cpu().data.item())
        rpn_stats.append(rpn_stat)
        rcnn_stats.append(rcnn_stat)
        lhi_stats.append(lhi_stat)
        mask_stats.append(mask_stat)

        #del input, 
        #del truth_box, truth_label
        #del net.rpn_proposals, net.detections
        #del net.total_loss, net.rpn_cls_loss, net.rpn_reg_loss, net.rcnn_cls_loss, net.rcnn_reg_loss, net.mask_loss, net.lhi_cls_loss, net.lhi_reg_loss
        #del net.rpn_logits_flat, net.rpn_deltas_flat

        #if net.use_rcnn:
        #    del net.rcnn_logits, net.rcnn_deltas
        #if net.use_mask:
        #    del net.mask_probs, net.mask_targets, net.mask_crop_boxes
        #if net.use_lhi:
        #    del net.rcnn_proposals, net.lhi_logits, net.lhi_deltas, net.lhi_crop_boxes

#        torch.cuda.empty_cache()

    rpn_stats = np.asarray(rpn_stats, np.float32)
    
    print('Train Epoch %d, iter %d, total time %f, loss %f' % (epoch, j, time.time() - s, np.average(total_loss)))
    #print('rpn_cls %f, rpn_reg %f, rcnn_cls %f, rcnn_reg %f, mask_loss %f' % \
    #    (np.average(rpn_cls_loss), np.average(rpn_reg_loss),
    #        np.average(rcnn_cls_loss), np.average(rcnn_reg_loss),
    #        np.average(mask_loss)))
    print('rpn_cls %f, rpn_reg %f, rcnn_cls %f, rcnn_reg %f, lhi_cls %f, lhi_reg %f, mask_loss %f' % \
        (np.average(rpn_cls_loss), np.average(rpn_reg_loss),
            np.average(rcnn_cls_loss), np.average(rcnn_reg_loss),
            np.average(lhi_cls_loss), np.average(lhi_reg_loss),
            np.average(mask_loss)))
    print('rpn_stats: tpr %f, tnr %f, total pos %d, total neg %d, reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
        100.0 * np.sum(rpn_stats[:, 0]) / np.sum(rpn_stats[:, 1]),
        100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3]),
        np.sum(rpn_stats[:, 1]),
        np.sum(rpn_stats[:, 3]),
        np.mean(rpn_stats[:, 4]),
        np.mean(rpn_stats[:, 5]),
        np.mean(rpn_stats[:, 6]),
        np.mean(rpn_stats[:, 7]),
        np.mean(rpn_stats[:, 8]),
        np.mean(rpn_stats[:, 9])))

    # Write to tensorboard
    writer.add_scalar('loss', np.average(total_loss), epoch)
    writer.add_scalar('rpn_cls', np.average(rpn_cls_loss), epoch)
    writer.add_scalar('rpn_reg', np.average(rpn_reg_loss), epoch)
    writer.add_scalar('rcnn_cls', np.average(rcnn_cls_loss), epoch)
    writer.add_scalar('rcnn_reg', np.average(rcnn_reg_loss), epoch)
    writer.add_scalar('lhi_cls', np.average(lhi_cls_loss), epoch)
    writer.add_scalar('lhi_reg', np.average(lhi_reg_loss), epoch)
    writer.add_scalar('mask_loss', np.average(mask_loss), epoch)

    writer.add_scalar('rpn_reg_z', np.mean(rpn_stats[:, 4]), epoch)
    writer.add_scalar('rpn_reg_y', np.mean(rpn_stats[:, 5]), epoch)
    writer.add_scalar('rpn_reg_x', np.mean(rpn_stats[:, 6]), epoch)
    writer.add_scalar('rpn_reg_d', np.mean(rpn_stats[:, 7]), epoch)
    writer.add_scalar('rpn_reg_h', np.mean(rpn_stats[:, 8]), epoch)
    writer.add_scalar('rpn_reg_w', np.mean(rpn_stats[:, 9]), epoch)

    if net.use_rcnn:
        confusion_matrix = np.asarray([stat[-1] for stat in rcnn_stats], np.int32)
        rcnn_stats = np.asarray([stat[:-1] for stat in rcnn_stats], np.float32)
        
        confusion_matrix = np.sum(confusion_matrix, 0)

        print('rcnn_stats: reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            np.mean(rcnn_stats[:, 0]),
            np.mean(rcnn_stats[:, 1]),
            np.mean(rcnn_stats[:, 2]),
            np.mean(rcnn_stats[:, 3]),
            np.mean(rcnn_stats[:, 4]),
            np.mean(rcnn_stats[:, 5])))
        # print_confusion_matrix(confusion_matrix)
        writer.add_scalar('rcnn_reg_z', np.mean(rcnn_stats[:, 0]), epoch)
        writer.add_scalar('rcnn_reg_y', np.mean(rcnn_stats[:, 1]), epoch)
        writer.add_scalar('rcnn_reg_x', np.mean(rcnn_stats[:, 2]), epoch)
        writer.add_scalar('rcnn_reg_d', np.mean(rcnn_stats[:, 3]), epoch)
        writer.add_scalar('rcnn_reg_h', np.mean(rcnn_stats[:, 4]), epoch)
        writer.add_scalar('rcnn_reg_w', np.mean(rcnn_stats[:, 5]), epoch)
    
    if net.use_lhi:
        lhi_stats = np.asarray([stat[:-1] for stat in lhi_stats], np.float32)
        
        print('lhi_stats: reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            np.mean(lhi_stats[:, 0]),
            np.mean(lhi_stats[:, 1]),
            np.mean(lhi_stats[:, 2]),
            np.mean(lhi_stats[:, 3]),
            np.mean(lhi_stats[:, 4]),
            np.mean(lhi_stats[:, 5])))
        writer.add_scalar('lhi_reg_z', np.mean(lhi_stats[:, 0]), epoch)
        writer.add_scalar('lhi_reg_y', np.mean(lhi_stats[:, 1]), epoch)
        writer.add_scalar('lhi_reg_x', np.mean(lhi_stats[:, 2]), epoch)
        writer.add_scalar('lhi_reg_d', np.mean(lhi_stats[:, 3]), epoch)
        writer.add_scalar('lhi_reg_h', np.mean(lhi_stats[:, 4]), epoch)
        writer.add_scalar('lhi_reg_w', np.mean(lhi_stats[:, 5]), epoch)
        if net.lhi_crop_boxes is not None and len(net.lhi_crop_boxes) > 0:
            draw_img, lhi_img, crop_img = plot_bbox_and_lhi(input, net.lhi_crop_boxes, truth_box, net.crop_lhi, net.crop_ims)
            grid = torchvision.utils.make_grid(draw_img, nrow=3)
            writer.add_image('RCNN Pred.', grid, epoch)
            lhi_grid = torchvision.utils.make_grid(lhi_img, nrow=3, normalize=True)
            writer.add_image('LHI Img.', lhi_grid, epoch)
            #crop_grid = torchvision.utils.make_grid(crop_img, nrow=3)
            #writer.add_image('Cropped Img.', crop_grid, epoch)
        

    if net.use_mask:
        mask_stats = np.array(mask_stats)
        for i in range(len(mask_stats[0])):
            s_region = mask_stats[:, i]
            s_region = s_region[~np.isnan(s_region)]
            print(config['roi_names'][i], ': ', np.round(np.mean(s_region), 4), ', ',)
            writer.add_scalar('mask_%s' % (config['roi_names'][i]), np.round(np.mean(s_region), 4), epoch)
    print
    print
    

def validate(net, val_loader, epoch, writer):
    net.set_mode('valid')
    rpn_cls_loss, rpn_reg_loss = [], []
    rcnn_cls_loss, rcnn_reg_loss = [], []
    lhi_cls_loss, lhi_reg_loss = [], []
    mask_loss = []
    total_loss = []
    rpn_stats = []
    rcnn_stats = []
    lhi_stats = []
    mask_stats = []

    s = time.time()
    for j, (input, truth_box, truth_label, truth_mask, masks) in tqdm(enumerate(val_loader), total=len(val_loader), desc='Val %d' % epoch):
        with torch.no_grad():
            input = Variable(input).cuda()
            truth_box = np.array(truth_box)
            truth_label = np.array(truth_label)

            net(input, truth_box, truth_label, truth_mask, masks)
            loss, rpn_stat, rcnn_stat, mask_stat, lhi_stat = net.loss()

        rpn_cls_loss.append(net.rpn_cls_loss.cpu().data.item())
        rpn_reg_loss.append(net.rpn_reg_loss.cpu().data.item())
        rcnn_cls_loss.append(net.rcnn_cls_loss.cpu().data.item())
        rcnn_reg_loss.append(net.rcnn_reg_loss.cpu().data.item())
        lhi_cls_loss.append(net.lhi_cls_loss.cpu().data.item())
        lhi_reg_loss.append(net.lhi_reg_loss.cpu().data.item())
        mask_loss.append(net.mask_loss.cpu().data.item())
        total_loss.append(loss.cpu().data.item())
        rpn_stats.append(rpn_stat)
        rcnn_stats.append(rcnn_stat)
        lhi_stats.append(lhi_stat)
        mask_stats.append(mask_stat)

    rpn_stats = np.asarray(rpn_stats, np.float32)
    print('Val Epoch %d, iter %d, total time %f, loss %f' % (epoch, j, time.time()-s, np.average(total_loss)))
    #print('rpn_cls %f, rpn_reg %f, rcnn_cls %f, rcnn_reg %f, mask_loss %f' % \
    #    (np.average(rpn_cls_loss), np.average(rpn_reg_loss),
    #        np.average(rcnn_cls_loss), np.average(rcnn_reg_loss),
    #        np.average(mask_loss)))
    print('rpn_cls %f, rpn_reg %f, rcnn_cls %f, rcnn_reg %f, lhi_cls %f, lhi_reg %f, mask_loss %f' % \
        (np.average(rpn_cls_loss), np.average(rpn_reg_loss),
            np.average(rcnn_cls_loss), np.average(rcnn_reg_loss),
            np.average(lhi_cls_loss), np.average(lhi_reg_loss),
            np.average(mask_loss)))
    print('rpn_stats: tpr %f, tnr %f, total pos %d, total neg %d, reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
        100.0 * np.sum(rpn_stats[:, 0]) / np.sum(rpn_stats[:, 1]),
        100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3]),
        np.sum(rpn_stats[:, 1]),
        np.sum(rpn_stats[:, 3]),
        np.mean(rpn_stats[:, 4]),
        np.mean(rpn_stats[:, 5]),
        np.mean(rpn_stats[:, 6]),
        np.mean(rpn_stats[:, 7]),
        np.mean(rpn_stats[:, 8]),
        np.mean(rpn_stats[:, 9])))
    
    # Write to tensorboard
    writer.add_scalar('loss', np.average(total_loss), epoch)
    writer.add_scalar('rpn_cls', np.average(rpn_cls_loss), epoch)
    writer.add_scalar('rpn_reg', np.average(rpn_reg_loss), epoch)
    writer.add_scalar('rcnn_cls', np.average(rcnn_cls_loss), epoch)
    writer.add_scalar('rcnn_reg', np.average(rcnn_reg_loss), epoch)
    writer.add_scalar('lhi_cls', np.average(lhi_cls_loss), epoch)
    writer.add_scalar('lhi_reg', np.average(lhi_reg_loss), epoch)
    writer.add_scalar('mask_loss', np.average(mask_loss), epoch)

    writer.add_scalar('rpn_reg_z', np.mean(rpn_stats[:, 4]), epoch)
    writer.add_scalar('rpn_reg_y', np.mean(rpn_stats[:, 5]), epoch)
    writer.add_scalar('rpn_reg_x', np.mean(rpn_stats[:, 6]), epoch)
    writer.add_scalar('rpn_reg_d', np.mean(rpn_stats[:, 7]), epoch)
    writer.add_scalar('rpn_reg_h', np.mean(rpn_stats[:, 8]), epoch)
    writer.add_scalar('rpn_reg_w', np.mean(rpn_stats[:, 9]), epoch)

    if net.use_rcnn:
        confusion_matrix = np.asarray([stat[-1] for stat in rcnn_stats], np.int32)
        rcnn_stats = np.asarray([stat[:-1] for stat in rcnn_stats], np.float32)
        
        confusion_matrix = np.sum(confusion_matrix, 0)
        print('rcnn_stats: reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            np.mean(rcnn_stats[:, 0]),
            np.mean(rcnn_stats[:, 1]),
            np.mean(rcnn_stats[:, 2]),
            np.mean(rcnn_stats[:, 3]),
            np.mean(rcnn_stats[:, 4]),
            np.mean(rcnn_stats[:, 5])))
        # print_confusion_matrix(confusion_matrix)
        writer.add_scalar('rcnn_reg_z', np.mean(rcnn_stats[:, 0]), epoch)
        writer.add_scalar('rcnn_reg_y', np.mean(rcnn_stats[:, 1]), epoch)
        writer.add_scalar('rcnn_reg_x', np.mean(rcnn_stats[:, 2]), epoch)
        writer.add_scalar('rcnn_reg_d', np.mean(rcnn_stats[:, 3]), epoch)
        writer.add_scalar('rcnn_reg_h', np.mean(rcnn_stats[:, 4]), epoch)
        writer.add_scalar('rcnn_reg_w', np.mean(rcnn_stats[:, 5]), epoch)
    
    if net.use_lhi:
        lhi_stats = np.asarray([stat[:-1] for stat in lhi_stats], np.float32)
        
        print('lhi_stats: reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            np.mean(lhi_stats[:, 0]),
            np.mean(lhi_stats[:, 1]),
            np.mean(lhi_stats[:, 2]),
            np.mean(lhi_stats[:, 3]),
            np.mean(lhi_stats[:, 4]),
            np.mean(lhi_stats[:, 5])))
        writer.add_scalar('lhi_reg_z', np.mean(lhi_stats[:, 0]), epoch)
        writer.add_scalar('lhi_reg_y', np.mean(lhi_stats[:, 1]), epoch)
        writer.add_scalar('lhi_reg_x', np.mean(lhi_stats[:, 2]), epoch)
        writer.add_scalar('lhi_reg_d', np.mean(lhi_stats[:, 3]), epoch)
        writer.add_scalar('lhi_reg_h', np.mean(lhi_stats[:, 4]), epoch)
        writer.add_scalar('lhi_reg_w', np.mean(lhi_stats[:, 5]), epoch)
        if net.lhi_crop_boxes is not None and len(net.lhi_crop_boxes) > 0:
            draw_img, lhi_img, crop_img = plot_bbox_and_lhi(input, net.lhi_crop_boxes, truth_box, net.crop_lhi, net.crop_ims)
            grid = torchvision.utils.make_grid(draw_img, nrow=3)
            writer.add_image('RCNN Pred.', grid, epoch)
            lhi_grid = torchvision.utils.make_grid(lhi_img, nrow=3, normalize=True)
            writer.add_image('LHI Img.', lhi_grid, epoch)
            #crop_grid = torchvision.utils.make_grid(crop_img, nrow=3)
            #writer.add_image('Cropped Img.', crop_grid, epoch)
        
    if net.use_mask:
        mask_stats = np.array(mask_stats)
        for i in range(len(mask_stats[0])):
            s_region = mask_stats[:, i]
            s_region = s_region[~np.isnan(s_region)]
            print(config['roi_names'][i], ': ', np.round(np.mean(s_region), 4), ', ',)
            writer.add_scalar('mask_%s' % (config['roi_names'][i]), np.round(np.mean(s_region), 4), epoch)
    print
    print
    
    del input, truth_box, truth_label
    del net.rpn_proposals, net.detections
    del net.total_loss, net.rpn_cls_loss, net.rpn_reg_loss, net.rcnn_cls_loss, net.rcnn_reg_loss, net.mask_loss, net.lhi_cls_loss, net.lhi_reg_loss

    if net.use_rcnn:
        del net.rcnn_logits, net.rcnn_deltas
    if net.use_mask:
        del net.mask_probs, net.mask_targets, net.mask_crop_boxes
    if net.use_lhi:
        del net.rcnn_proposals, net.lhi_logits, net.lhi_deltas, net.lhi_crop_boxes
    
    torch.cuda.empty_cache()

def print_confusion_matrix(confusion_matrix):
    line_new = '{:>4}  ' * (len(config['roi_names']) + 2)
    print(line_new.format('gt/p', *list(range(len(config['roi_names']) + 1))))

    for i in range(len(config['roi_names']) + 1):
        print(line_new.format(i, *list(confusion_matrix[i])))
        

if __name__ == '__main__':
    main()



