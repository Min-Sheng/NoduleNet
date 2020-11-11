from net.layer.util import box_transform, box_transform_inv, clip_boxes
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Variable
try:
    from utils.pybox import *
except ImportError:
    print('Warning: C++ module import failed! This should only happen in deployment')
    from utils.util import py_nms as torch_nms
    from utils.util import py_box_overlap as torch_overlap


def lhi_encode(window, truth_box, weight):
    return box_transform(window, truth_box, weight)


def lhi_decode(window, delta, weight):
    return  box_transform_inv(window, delta, weight)

def get_probability_lhi(cfg, mode, inputs, proposals, logits, deltas):
    if mode in ['train',]:
        nms_pre_score_threshold = cfg['lhi_train_nms_pre_score_threshold']
        nms_overlap_threshold   = cfg['lhi_train_nms_overlap_threshold']

    elif mode in ['valid', 'test','eval']:
        nms_pre_score_threshold = cfg['lhi_test_nms_pre_score_threshold']
        nms_overlap_threshold   = cfg['lhi_test_nms_overlap_threshold']
    else:
        raise ValueError('lhi_nms(): invalid mode = %s?'%mode)

    num_class = cfg['num_class']
    probs = F.softmax(logits).cpu().data.numpy()
    deltas = deltas.cpu().data.numpy().reshape(-1, num_class, 6)
    proposals = proposals.cpu().data.numpy()

    for j in range(1, num_class):  # skip background
        idx = np.where(probs[:, j] > nms_pre_score_threshold)[0]
        if len(idx) > 0:
            p = probs[idx, j].reshape(-1, 1)
            d = deltas[idx, j]
            box = lhi_decode(proposals [idx, 2:8], d, cfg['box_reg_weight'])
            box = clip_boxes(box, inputs.shape[2:])
            js = np.expand_dims(np.array([j] * len(p)), axis=-1)
            output = np.concatenate((p, box, js), 1)

    return torch.from_numpy(output).cuda().float()
