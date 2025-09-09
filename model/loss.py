# -*- coding:utf8 -*-

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import bbox_iou, xyxy2xywh
from .darknet import ConvBatchNormReLU
import cv2

class InfoNCE(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.proj_A = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(embed_dim, embed_dim))
        self.proj_B = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)),\
                                    ConvBatchNormReLU(embed_dim, embed_dim, 1, 1, 0, 1, leaky=True))
        self.maxpool = torch.nn.AdaptiveMaxPool1d(1)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, featA, featB):
        B, C, _, _ = featA.shape
        featA = self.proj_A(featA)
        featB = self.proj_B(featB)
        
        B, C, _, _ = featB.shape
        featB = featB.view(B, C, -1).permute(2,1,0) # B,C,G->G,C,B
        featA = featA.unsqueeze(0) # 1,C,B      

        featA = F.normalize(featA, dim=1)
        featB = F.normalize(featB, dim=1)
        
        logits_A = self.logit_scale * featA @ featB # [1,C,B] * [G,C,B] -> [G,B,B]
        logits_A = logits_A.permute(1,2,0) # [G,B,B] -> [B,B,G]
        logits_A = self.maxpool(logits_A).squeeze()
        logits_B = logits_A.T

        labels = torch.arange(len(logits_A), dtype=torch.long, device=featA.device)
        loss = (self.loss_function(logits_A, labels) + self.loss_function(logits_B, labels)) / 2

        return loss


def adjust_learning_rate(args, optimizer, i_iter):
    
    lr = args.lr*((0.1)**(i_iter//10))
        
    print(("lr", lr))
    for param_idx, param in enumerate(optimizer.param_groups):
        param['lr'] = lr

# the shape of the target is (batch_size, anchor_count, 5, grid_wh, grid_wh)
def yolo_loss(predictions, gt_bboxes, anchors_full, best_anchor_gi_gj, image_wh):
    batch_size, grid_stride = predictions.shape[0], image_wh // predictions.shape[3]
    best_anchor, gi, gj = best_anchor_gi_gj[:, 0], best_anchor_gi_gj[:, 1], best_anchor_gi_gj[:, 2]
    scaled_anchors = anchors_full / grid_stride
    mseloss = torch.nn.MSELoss(size_average=True)
    celoss_confidence = torch.nn.CrossEntropyLoss(size_average=True)
    #celoss_cls = torch.nn.CrossEntropyLoss(size_average=True)

    selected_predictions = predictions[range(batch_size), best_anchor, :, gj, gi]

    #---bbox loss---
    pred_bboxes = torch.zeros_like(gt_bboxes)
    pred_bboxes[:, 0:2] = selected_predictions[:, 0:2].sigmoid()
    pred_bboxes[:, 2:4] = selected_predictions[:, 2:4]
    
    loss_x = mseloss(pred_bboxes[:,0], gt_bboxes[:,0])
    loss_y = mseloss(pred_bboxes[:,1], gt_bboxes[:,1])
    loss_w = mseloss(pred_bboxes[:,2], gt_bboxes[:,2])
    loss_h = mseloss(pred_bboxes[:,3], gt_bboxes[:,3])

    loss_bbox = loss_x + loss_y + loss_w + loss_h

    #---confidence loss---
    pred_confidences = predictions[:,:,4,:,:]
    gt_confidences = torch.zeros_like(pred_confidences)
    gt_confidences[range(batch_size), best_anchor, gj, gi] = 1
    pred_confidences, gt_confidences = pred_confidences.reshape(batch_size, -1), \
                    gt_confidences.reshape(batch_size, -1)
    loss_confidence = celoss_confidence(pred_confidences, gt_confidences.max(1)[1])

    return loss_bbox, loss_confidence

def yolo_rotate_loss(predictions, gt_bboxes, anchors_full, best_anchor_gi_gj, image_wh):
    batch_size, grid_stride = predictions.shape[0], image_wh // predictions.shape[3]
    best_anchor, gi, gj = best_anchor_gi_gj[:, 0], best_anchor_gi_gj[:, 1], best_anchor_gi_gj[:, 2]
    scaled_anchors = anchors_full / grid_stride
    mseloss = torch.nn.MSELoss(size_average=True)
    celoss_confidence = torch.nn.CrossEntropyLoss(size_average=True)
    #celoss_cls = torch.nn.CrossEntropyLoss(size_average=True)

    selected_predictions = predictions[range(batch_size), best_anchor, :, gj, gi]

    #---bbox loss---
    pred_bboxes = torch.zeros_like(gt_bboxes)
    pred_bboxes[:, 0:2] = selected_predictions[:, 0:2].sigmoid()
    pred_bboxes[:, 2:4] = selected_predictions[:, 2:4]
    pred_bboxes[:, 4] = selected_predictions[:, 4]
    
    
    loss_x = mseloss(pred_bboxes[:,0], gt_bboxes[:,0])
    loss_y = mseloss(pred_bboxes[:,1], gt_bboxes[:,1])
    loss_w = mseloss(pred_bboxes[:,2], gt_bboxes[:,2])
    loss_h = mseloss(pred_bboxes[:,3], gt_bboxes[:,3])
    loss_t = mseloss(pred_bboxes[:,4], gt_bboxes[:,4])

    loss_bbox = loss_x + loss_y + loss_w + loss_h + 2*loss_t

    #---confidence loss---
    pred_confidences = predictions[:,:,5,:,:]
    gt_confidences = torch.zeros_like(pred_confidences)
    gt_confidences[range(batch_size), best_anchor, gj, gi] = 1
    pred_confidences, gt_confidences = pred_confidences.reshape(batch_size, -1), \
                    gt_confidences.reshape(batch_size, -1)
    loss_confidence = celoss_confidence(pred_confidences, gt_confidences.max(1)[1])

    return loss_bbox, loss_confidence


#target_coord:batch_size, 5
def build_target(ori_gt_bboxes, anchors_full, image_wh, grid_wh):
    #the default value of coord_dim is 5
    batch_size, coord_dim, grid_stride, anchor_count = ori_gt_bboxes.shape[0], ori_gt_bboxes.shape[1], image_wh//grid_wh, anchors_full.shape[0]
    
    gt_bboxes = xyxy2xywh(ori_gt_bboxes)
    gt_bboxes = (gt_bboxes/image_wh) * grid_wh
    scaled_anchors = anchors_full/grid_stride

    gxy = gt_bboxes[:, 0:2]
    gwh = gt_bboxes[:, 2:4]
    gij = gxy.long()

    #get the best anchor for each target bbox
    gt_bboxes_tmp, scaled_anchors_tmp = torch.zeros_like(gt_bboxes), torch.zeros((anchor_count, coord_dim), device=gt_bboxes.device)
    gt_bboxes_tmp[:, 2:4] = gwh
    gt_bboxes_tmp = gt_bboxes_tmp.unsqueeze(1).repeat(1, anchor_count, 1).view(-1, coord_dim)
    scaled_anchors_tmp[:, 2:4] = scaled_anchors
    scaled_anchors_tmp = scaled_anchors_tmp.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, coord_dim)
    anchor_ious = bbox_iou(gt_bboxes_tmp, scaled_anchors_tmp).view(batch_size, -1)
    best_anchor=torch.argmax(anchor_ious, dim=1)
    
    twh = torch.log(gwh / scaled_anchors[best_anchor] + 1e-16)
    #print((gxy.dtype, gij.dtype, twh.dtype, gwh.dtype, scaled_anchors.dtype, 'inner'))
    #print((gxy.shape, gij.shape, twh.shape, gwh.shape), flush=True)
    #print(('gxy,gij,twh', gxy, gij, twh), flush=True)
    return torch.cat((gxy - gij, twh), 1), torch.cat((best_anchor.unsqueeze(1), gij), 1)

def rbox_overlaps(boxes, query_boxes, indicator=None, thresh=1e-1):
    # rewrited by cython
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    a_tt = boxes[:, 4]
    a_ws = boxes[:, 2] - boxes[:, 0]
    a_hs = boxes[:, 3] - boxes[:, 1]
    a_xx = boxes[:, 0] + a_ws * 0.5
    a_yy = boxes[:, 1] + a_hs * 0.5
    

    b_tt = query_boxes[:, 4]
    b_ws = query_boxes[:, 2] - query_boxes[:, 0]
    b_hs = query_boxes[:, 3] - query_boxes[:, 1]
    b_xx = query_boxes[:, 0] + b_ws * 0.5
    b_yy = query_boxes[:, 1] + b_hs * 0.5

    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        box_area = b_ws[k] * b_hs[k]
        for n in range(N):
            if indicator is not None and indicator[n, k] < thresh:
                continue
            ua = a_ws[n] * a_hs[n] + box_area
            rtn, contours = cv2.rotatedRectangleIntersection(
                ((a_xx[n], a_yy[n]), (a_ws[n], a_hs[n]), a_tt[n]),
                ((b_xx[k], b_yy[k]), (b_ws[k], b_hs[k]), b_tt[k])
            )
            if rtn == 1:
                ia = cv2.contourArea(contours)
                overlaps[n, k] = ia / (ua - ia)
            elif rtn == 2:
                ia = np.minimum(ua - box_area, box_area)
                overlaps[n, k] = ia / (ua - ia)
    return overlaps

def build_rotate_target(ori_gt_bboxes, anchors_full, image_wh, grid_wh):
    #the default value of coord_dim is 5
    batch_size, coord_dim, grid_stride, anchor_count = ori_gt_bboxes.shape[0], ori_gt_bboxes.shape[1], image_wh//grid_wh, anchors_full.shape[0]
    
    gt_bboxes = (ori_gt_bboxes/image_wh) * grid_wh
    scaled_anchors = anchors_full/grid_stride

    gxy = gt_bboxes[:, 0:2]
    gwh = gt_bboxes[:, 2:4]
    gtheta = ori_gt_bboxes[:, 4]
    gij = gxy.long()

    #get the best anchor for each target bbox
    gt_bboxes_tmp, scaled_anchors_tmp = torch.zeros_like(gt_bboxes), torch.zeros((anchor_count, coord_dim), device=gt_bboxes.device)
    gt_bboxes_tmp[:, 2:4] = gwh
    gt_bboxes_tmp[:, 4] = gtheta
  

    scaled_anchors_tmp[:, 2:4] = scaled_anchors
    scaled_anchors_tmp[:, 4] = torch.where(scaled_anchors_tmp[:, 2] > scaled_anchors_tmp[:, 3],
    torch.tensor(0.0, device=scaled_anchors_tmp.device),
    torch.tensor(-90.0, device=scaled_anchors_tmp.device))
    record_scaled_anchor = scaled_anchors_tmp.clone()
  
    ious = rbox_overlaps(
                gt_bboxes_tmp.cpu().numpy(),
                scaled_anchors_tmp.cpu().numpy()
                )
    if not torch.is_tensor(ious):
        ious = torch.from_numpy(ious).to(device=gt_bboxes.device)
    
    anchor_ious = ious.view(batch_size, -1)
    # print(anchor_ious)
    best_anchor=torch.argmax(anchor_ious, dim=1)
    # print(best_anchor)
   
    
    twh = torch.log(gwh / scaled_anchors[best_anchor] + 1e-16)
    
    ttheta = gtheta - record_scaled_anchor[best_anchor][:, 4]
    ttheta = ttheta.unsqueeze(1)/180
    # print('twh',twh)
    # print('ttheta', ttheta)
    # print('ttheta', twh.size())
    #print((gxy.dtype, gij.dtype, twh.dtype, gwh.dtype, scaled_anchors.dtype, 'inner'))
    #print((gxy.shape, gij.shape, twh.shape, gwh.shape), flush=True)
    #print(('gxy,gij,twh', gxy, gij, twh), flush=True)
    return torch.cat((gxy - gij, twh, ttheta), 1), torch.cat((best_anchor.unsqueeze(1), gij), 1)