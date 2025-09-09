# -*- coding:utf8 -*-

import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

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

def xyxy2xywh(x):  # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    #y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y = torch.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

def xywh2xyxy(x):  # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    #y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y = torch.zeros_like(x)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y
    
def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # print(box1, box1.shape)
    # print(box2, box2.shape)
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def multiclass_metrics(pred, gt):
  """
  check precision and recall for predictions.
  Output: overall = {precision, recall, f1}
  """
  eps=1e-6
  overall = {'precision': -1, 'recall': -1, 'f1': -1}
  NP, NR, NC = 0, 0, 0  # num of pred, num of recall, num of correct
  for ii in range(pred.shape[0]):
    pred_ind = np.array(pred[ii]>0.5, dtype=int)
    gt_ind = np.array(gt[ii]>0.5, dtype=int)
    inter = pred_ind * gt_ind
    # add to overall
    NC += np.sum(inter)
    NP += np.sum(pred_ind)
    NR += np.sum(gt_ind)
  if NP > 0:
    overall['precision'] = float(NC)/NP
  if NR > 0:
    overall['recall'] = float(NC)/NR
  if NP > 0 and NR > 0:
    overall['f1'] = 2*overall['precision']*overall['recall']/(overall['precision']+overall['recall']+eps)
  return overall

def plot_bboxes_cv(image_path, preds, color, filename):
    """
    从指定路径读取图像，绘制预测框和真值框，并保存为新图片.

    参数:
    image_path (str): 输入图像的文件路径
    preds (torch.Tensor): 预测框, 形状为 (N, 4) 的张量，表示 (x1, y1, x2, y2)
    truths (torch.Tensor): 真值框, 形状为 (M, 4) 的张量，表示 (x1, y1, x2, y2)
    filename (str): 保存图片的文件名
    """

    # 读取输入图像
    print(image_path)
    image = cv2.imread(image_path)

    # 将预测框绘制为黄色
    for box in preds:
        x1, y1, x2, y2 = (box.cpu().numpy()).astype(int)  # 转到CPU并转换为整数
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)  # BGR格式：黄色

    # 将真值框绘制为红色
    # for box in truths:
    #     x1, y1, x2, y2 = (box.cpu().numpy()).astype(int)  # 转到CPU并转换为整数
    #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # BGR格式：红色

    # 保存图像
    cv2.imwrite(filename, image)




def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

#target_gj : target_bbox[:, :, :, x, :]
#target_gi : target_bbox[:, :, :, :, x]
def eval_iou_acc(pred_anchor, target_bbox, anchors_full, target_gi, target_gj, image_wh, iou_threshold_list):
# def eval_iou_acc(pred_anchor, target_bbox, anchors_full, target_gi, target_gj, image_wh, iou_threshold_list, queryimg_name, rsimg_name):
    #print(pred_anchor)

    batch_size, grid_stride = target_bbox.shape[0], image_wh // pred_anchor.shape[3]
    #batch_size, anchor_count, xywh+confidence, grid_height, grid_width
    assert(len(pred_anchor.shape) == 5)
    assert(pred_anchor.shape[3] == pred_anchor.shape[4])
    
    ## eval: convert center+offset to box prediction
    ## calculate at rescaled image during validation for speed-up
    pred_confidence = pred_anchor[:,:,4,:,:]
    scaled_anchors = anchors_full / grid_stride
    
    pred_gi, pred_gj = torch.zeros_like(target_gi), torch.zeros_like(target_gj)
    pred_bbox = torch.zeros_like(target_bbox)
    for batch_idx in range(batch_size):
        best_n, gj, gi = torch.where(pred_confidence[batch_idx].max() == pred_confidence[batch_idx])
        best_n, gj, gi = best_n[0], gj[0], gi[0]
        pred_gj[batch_idx], pred_gi[batch_idx] = gj, gi
        #print((best_n, gi, gj))
        
        pred_bbox[batch_idx, 0] = pred_anchor[batch_idx, best_n, 0, gj, gi].sigmoid() + gi
        pred_bbox[batch_idx, 1] = pred_anchor[batch_idx, best_n, 1, gj, gi].sigmoid() + gj
        pred_bbox[batch_idx, 2] = torch.exp(pred_anchor[batch_idx, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
        pred_bbox[batch_idx, 3] = torch.exp(pred_anchor[batch_idx, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
    pred_bbox = pred_bbox * grid_stride
    pred_bbox = xywh2xyxy(pred_bbox)
    
    #可视化---------------------------
    # sat_image_root = "/home/zzh/TGRS_PROJECT/DetGeo_bbox/Visual_Hbox/uav2sat_dark/"
    # satname = sat_image_root + rsimg_name[0]
    # savename = '/home/zzh/TGRS_PROJECT/zzh_DetGeo_TGRS_bbox/visual_Hbox_all/uav2sat_dark/' + os.path.splitext(queryimg_name[0])[0] + '__________' + rsimg_name[0]
    # plot_bboxes_cv(satname, pred_bbox, color=(0,255,255), filename=savename)
    #可视化--------------------------

    # === Center Distance Normalization ===
    # 取预测框和真值框中心点坐标
    pred_cx = (pred_bbox[:, 0] + pred_bbox[:, 2]) / 2  # [B]
    pred_cy = (pred_bbox[:, 1] + pred_bbox[:, 3]) / 2
    gt_cx = (target_bbox[:, 0] + target_bbox[:, 2]) / 2
    gt_cy = (target_bbox[:, 1] + target_bbox[:, 3]) / 2

    # 欧氏距离
    dist = torch.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2)  # [B]

    # 图像对角线归一化
    norm_factor = (image_wh ** 2 + image_wh ** 2) ** 0.5
    norm_dist = (dist / norm_factor).mean()

    ## box iou
    iou = bbox_iou(pred_bbox, target_bbox, x1y1x2y2=True)
    each_acc50 = iou>0.5
    accu_list, each_acc_list=list(), list()
    for threshold in iou_threshold_list:
        each_acc = iou>threshold
        accu = torch.sum(each_acc)/batch_size
        accu_list.append(accu)
        each_acc_list.append(each_acc)
    accu_center = torch.sum((target_gi == pred_gi) * (target_gj == pred_gj))/batch_size
    iou = torch.sum(iou)/batch_size

    return accu_list, accu_center, iou, each_acc_list, pred_bbox, target_bbox, norm_dist

def rbox_overlaps_one2one(boxes, query_boxes, indicator=None, thresh=1e-1):
    # rewrited by cython
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    a_tt = boxes[:, 4]
    a_ws = boxes[:, 2] 
    a_hs = boxes[:, 3]
    a_xx = boxes[:, 0] 
    a_yy = boxes[:, 1] 
    
    
    b_tt = query_boxes[:, 4]
    b_ws = query_boxes[:, 2]
    b_hs = query_boxes[:, 3]
    b_xx = query_boxes[:, 0] 
    b_yy = query_boxes[:, 1] 

    overlaps = np.zeros(N, dtype=np.float32)
    for k in range(K):
        box_area = b_ws[k] * b_hs[k]
        for n in range(N):
            if n!=k:
                continue
            if indicator is not None and indicator[n, k] < thresh:
                continue
            ua = a_ws[n] * a_hs[n] + box_area
            rtn, contours = cv2.rotatedRectangleIntersection(
                ((a_xx[n], a_yy[n]), (a_ws[n], a_hs[n]), a_tt[n]),
                ((b_xx[k], b_yy[k]), (b_ws[k], b_hs[k]), b_tt[k])
            )
            if rtn == 1:
                ia = cv2.contourArea(contours)
                overlaps[n] = ia / (ua - ia)
            elif rtn == 2:
                ia = np.minimum(ua - box_area, box_area)
                overlaps[n] = ia / (ua - ia)
    return overlaps

def eval_rotate_iou_acc(pred_anchor, target_bbox, anchors_full, target_gi, target_gj, image_wh, iou_threshold_list):
# def eval_iou_acc(pred_anchor, target_bbox, anchors_full, target_gi, target_gj, image_wh, iou_threshold_list, queryimg_name, rsimg_name):
    #print(pred_anchor)

    batch_size, grid_stride = target_bbox.shape[0], image_wh // pred_anchor.shape[3]
    #batch_size, anchor_count, xywh+confidence, grid_height, grid_width
    assert(len(pred_anchor.shape) == 5)
    assert(pred_anchor.shape[3] == pred_anchor.shape[4])
    
    ## eval: convert center+offset to box prediction
    ## calculate at rescaled image during validation for speed-up
    pred_confidence = pred_anchor[:,:,5,:,:]
    scaled_anchors = anchors_full / grid_stride

    theta_anchors = torch.where(scaled_anchors[:, 0] > scaled_anchors[:, 1],
    torch.tensor(0.0, device=scaled_anchors.device),
    torch.tensor(-90.0, device=scaled_anchors.device))
    # print(theta_anchors)
    
    pred_gi, pred_gj = torch.zeros_like(target_gi), torch.zeros_like(target_gj)
    pred_bbox = torch.zeros_like(target_bbox)
    for batch_idx in range(batch_size):
        best_n, gj, gi = torch.where(pred_confidence[batch_idx].max() == pred_confidence[batch_idx])
        best_n, gj, gi = best_n[0], gj[0], gi[0]
        pred_gj[batch_idx], pred_gi[batch_idx] = gj, gi
        #print((best_n, gi, gj))
        
        pred_bbox[batch_idx, 0] = pred_anchor[batch_idx, best_n, 0, gj, gi].sigmoid() + gi
        pred_bbox[batch_idx, 1] = pred_anchor[batch_idx, best_n, 1, gj, gi].sigmoid() + gj
        pred_bbox[batch_idx, 2] = torch.exp(pred_anchor[batch_idx, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
        pred_bbox[batch_idx, 3] = torch.exp(pred_anchor[batch_idx, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
        pred_bbox[batch_idx, 4] = pred_anchor[batch_idx, best_n, 4, gj, gi]*180 + theta_anchors[best_n]
    pred_bbox_xywh = pred_bbox[:, :4] * grid_stride
    pred_bbox[:, :4] = pred_bbox_xywh

   
    
    # print(queryimg_name, rsimg_name)
    # sat_image_root = "/home/zzh/Desktop/zzh_DetGeo/visual/half_project/"
    # satname = sat_image_root + rsimg_name[0]
    # savename = 'visual/all/'+ queryimg_name[0][:-4]+ '_' + rsimg_name[0]
    # plot_bboxes_cv(satname, pred_bbox, target_bbox, savename)

    ## box iou
    # print('pred_bbox',pred_bbox)
    # print('target_bbox',target_bbox)
    iou = rbox_overlaps_one2one(
                pred_bbox.detach().cpu().numpy(),
                target_bbox.detach().cpu().numpy()
                )
    if not torch.is_tensor(iou):
        iou = torch.from_numpy(iou).to(device=pred_anchor.device)

    
    each_acc50 = iou>0.5
    accu_list, each_acc_list=list(), list()
    for threshold in iou_threshold_list:
        each_acc = iou>threshold
        accu = torch.sum(each_acc)/batch_size
        accu_list.append(accu)
        each_acc_list.append(each_acc)
    accu_center = torch.sum((target_gi == pred_gi) * (target_gj == pred_gj))/batch_size
    iou = torch.sum(iou)/batch_size

    return accu_list, accu_center, iou, each_acc_list, pred_bbox, target_bbox




def visualize_and_save_attention(attn_score, original_image_path, output_path):

    pic_root_path = "/home/zzh/Desktop/CVOGL/CVOGL_DroneAerial/satellite/"
    original_image_path = original_image_path[0]
    output_path = output_path + original_image_path
    original_image_path = pic_root_path + original_image_path
    # 1. 将 attn_score 从 GPU 移到 CPU
    attn_score = attn_score.cpu().detach().numpy()  # 转换为 NumPy 数组
    attn_score = attn_score.squeeze()  # 去掉多余的维度，形状变为 (64, 64)

    # 2. 将 attn_score 扩展到与原图相同的大小
    attn_score_resized = cv2.resize(attn_score, (1024, 1024), interpolation=cv2.INTER_LINEAR)

    # 3. 将 attn_score 转换为颜色图
    attn_colormap = plt.get_cmap('rainbow')(attn_score_resized)  # 获取彩虹色映射
    attn_colormap = attn_colormap[:, :, :3]  # 去掉 alpha 通道
    attn_colormap = (attn_colormap * 255).astype(np.uint8)  # 转换为 0-255 范围

    # 4. 读取原图并转换为 RGBA 格式
    original_image = cv2.imread(original_image_path)  # 读取原图像
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # 转换颜色通道顺序
    original_image = cv2.resize(original_image, (1024, 1024))  # 确保原图大小为 1024x1024

    # 5. 将颜色图与原图叠加
    alpha = 0.5  # 设置透明度
    overlay = cv2.addWeighted(original_image, 1 - alpha, attn_colormap, alpha, 0)
    
    # 6. 保存结果
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))  # 保存处理后的图像
