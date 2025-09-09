# -*- coding: utf-8 -*-

import os
import sys
import cv2
import random
import numpy as np
import torch
import copy
import time
from tqdm import tqdm
from torch.utils.data import Dataset
import albumentations
from shapely.geometry import Polygon
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import argparse





cv2.setNumThreads(0)
    
class DatasetNotFoundError(Exception):
    pass

class MyAugment:
    def __init__(self) -> None:
        self.transform = albumentations.Compose([
                albumentations.Blur(p=0.01),
                albumentations.MedianBlur(p=0.01),
                albumentations.ToGray(p=0.01),
                albumentations.CLAHE(p=0.01),
                albumentations.RandomBrightnessContrast(p=0.0),
                albumentations.RandomGamma(p=0.0),
                albumentations.ImageCompression(quality_lower=75, p=0.0)])
    
    def augment_hsv(self, im, hgain=0.5, sgain=0.5, vgain=0.5):
        # HSV color-space augmentation
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=im)  # no return needed

    def __call__(self, img, bbox):
        imgh,imgw, _ = img.shape
        x, y, w, h = (bbox[0]+bbox[2])/2/imgw, (bbox[1]+bbox[3])/2/imgh, (bbox[2]-bbox[0])/imgw, (bbox[3]-bbox[1])/imgh
        img = self.transform(image=img)['image']
        #self.augment_hsv(img)
        # Flip up-down
        if random.random() < 0.5:
            img = np.flipud(img)
            y = 1-y
            
        # Flip left-right
        if random.random() < 0.5:
            img = np.fliplr(img)
            x = 1-x
        #
        new_imgh, new_imgw, _ = img.shape
        assert new_imgh==imgh, new_imgw==imgw
        x, y, w, h = x*imgw, y*imgh, w*imgw, h*imgh

        # Crop image
        iscropped=False
        if random.random() < 0.5:
            left, top, right, bottom = x-w/2, y-h/2, x+w/2, y+h/2
            if left >= new_imgw/2:
                start_cropped_x = random.randint(0, int(0.15*new_imgw))
                img = img[:, start_cropped_x:, :]
                left, right = left - start_cropped_x, right - start_cropped_x
            if right <= new_imgw/2:
                start_cropped_x = random.randint(int(0.85*new_imgw), new_imgw)
                img = img[:, 0:start_cropped_x, :]
            if top >= new_imgh/2:
                start_cropped_y = random.randint(0, int(0.15*new_imgh))
                img = img[start_cropped_y:, :, :]
                top, bottom = top - start_cropped_y, bottom - start_cropped_y
            if bottom <= new_imgh/2:
                start_cropped_y = random.randint(int(0.85*new_imgh), new_imgh)
                img = img[0:start_cropped_y, :, :]
            cropped_imgh, cropped_imgw, _ = img.shape
            left, top, right, bottom = left/cropped_imgw, top/cropped_imgh, right/cropped_imgw, bottom/cropped_imgh
            if cropped_imgh != new_imgh or cropped_imgw != new_imgw:
                img = cv2.resize(img, (new_imgh, new_imgw))
            new_cropped_imgh, new_cropped_imgw, _ = img.shape
            left, top, right, bottom = left*new_cropped_imgw, top*new_cropped_imgh, right*new_cropped_imgw, bottom*new_cropped_imgh 
            x, y, w, h = (left+right)/2, (top+bottom)/2, right-left, bottom-top
            iscropped=True
        #if iscropped:
        #    print((new_imgw, new_imgh))
        #    print((cropped_imgw, cropped_imgh), flush=True)
        #    print('============')
        #print(type(img))
        #draw_bbox = np.array([x-w/2, y-h/2, x+w/2, y+h/2], dtype=int)
        #print(('draw_bbox', iscropped, draw_bbox), flush=True)
        #img_new=draw_rectangle(img, draw_bbox)
        #cv2.imwrite('tmp/'+str(random.randint(0,5000))+"_"+str(iscropped)+".jpg", img_new)

        new_bbox = [(x-w/2), y-h/2, x+w/2, y+h/2]
        #print(bbox)
        #print(new_bbox)
        #print('---end---')
        return img, np.array(new_bbox, dtype=int)

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, x_max, y_min, y_max = int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])
    print(bbox, flush=True)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img

class RSDataset(Dataset):
    def __init__(self, data_root, data_name='CVOGL', split_name='train', img_size=1024,
                 transform=None, augment=False):
        self.data_root = data_root
        self.data_name = data_name
        self.img_size = img_size
        self.transform = transform
        self.split_name = split_name
        self.augment=augment
        self.shuffle_batch_size = 32 #add

        self.myaugment = MyAugment()

        if self.data_name == 'CVOGL_DroneAerial':
            data_dir = os.path.join(data_root, self.data_name)
            data_path = os.path.join(data_dir, '{0}_{1}.pth'.format(self.data_name, split_name))
            # data_path = '/home/zzh/Desktop/CVOGL/CVOGL_DroneAerial/cls_ann/test/others.pth'
            self.data_list = torch.load(data_path)
            self.samples = list(range(len(self.data_list))) #add
            self.queryimg_dir = os.path.join(data_dir, 'query')
            self.rsimg_dir = os.path.join(data_dir, 'satellite')
            self.rs_wh = 1024
            self.query_featuremap_hw = (256, 256) #52 #32
        elif self.data_name == 'CVOGL_SVI':
            data_dir = os.path.join(data_root, self.data_name)
            data_path = os.path.join(data_dir, '{0}_{1}.pth'.format(self.data_name, split_name))
            # data_path = '/home/zzh/Desktop/CVOGL/CVOGL_SVI/cls_ann/test/others.pth'
            self.data_list = torch.load(data_path)
            self.samples = list(range(len(self.data_list))) #add
            self.queryimg_dir = os.path.join(data_dir, 'query')
            self.rsimg_dir = os.path.join(data_dir, 'satellite')
            self.rs_wh = 1024
            self.query_featuremap_hw = (256, 512)
        else:
            assert(False)
        
        self.rs_transform = albumentations.Compose([   
            albumentations.RandomSizedBBoxSafeCrop(width=self.rs_wh, height=self.rs_wh, erosion_rate=0.2, p=0.2),
	        albumentations.RandomRotate90(p=0.5),
	        albumentations.GaussNoise(p=0.5),
	        albumentations.HueSaturationValue(p=0.3),
	        albumentations.OneOf([
		        albumentations.Blur(p=0.4),
		        albumentations.MedianBlur(p=0.3),
	        ], p=0.5),
	        albumentations.OneOf([
		        albumentations.RandomBrightnessContrast(p=0.4),
		        albumentations.CLAHE(p=0.3),
	        ], p=0.5),
	        albumentations.ToGray(p=0.2),
	        albumentations.RandomGamma(p=0.3),], bbox_params=albumentations.BboxParams(format='pascal_voc'))

    def __len__(self):
        return len(self.data_list)

    def posemb_sincos_2d(self, h, w, dim, temperature: int = 10000, dtype=torch.float32):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
        omega = torch.arange(dim // 4) / (dim // 4 - 1)
        omega = 1.0 / (temperature ** omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        pe = pe.type(dtype)
        return pe.detach().numpy()

    def shuffle(self, sim_dict=None, neighbour_select=16, neighbour_range=32):
        '''
        custom shuffle function for unique class_id sampling in batch
        '''
        print("\nShuffle Dataset:")
        if sim_dict is None:
            random.shuffle(self.samples)
        else:
            index_pool = copy.deepcopy(self.samples)
            pairs_epoch = set()
            current_batch = []
            batches = []

            pbar = tqdm(total=len(index_pool))  # 初始化进度条

            while index_pool:
                pbar.update()
                idx = index_pool.pop(0)  # 从池中取出一个索引
                
                if idx not in pairs_epoch and len(current_batch) < self.shuffle_batch_size:
                    current_batch.append(idx)
                    pairs_epoch.add(idx)

                    if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                        near_similarity = copy.deepcopy(sim_dict[idx][:neighbour_range])

                        neighbour_split = neighbour_select // 2
                        
                        near_always = near_similarity[:neighbour_split]
                        near_random = near_similarity[neighbour_split:]
                        random.shuffle(near_random)
                        near_random = near_random[:neighbour_split]
                        near_similarity_select = near_always + near_random

                        for idx_near in near_similarity_select:
                            if len(current_batch) >= self.shuffle_batch_size:
                                break
                            if idx_near not in pairs_epoch:
                                current_batch.append(idx_near)
                                pairs_epoch.add(idx_near)   


                if len(current_batch) >= self.shuffle_batch_size:
                    batches.extend(current_batch)
                    current_batch = []
            pbar.close()

            time.sleep(0.3)
            
            # 检查并添加最后一个不完整的批次
            if current_batch:
                batches.extend(current_batch)

            self.samples = batches
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.samples), len(self.samples)))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0], self.samples[-1]))




    def __getitem__(self, idx):
        _, queryimg_name, rsimg_name, _, click_xy, bbox, _, cls_name = self.data_list[self.samples[idx]] #add self.samples
        
        ## box format: to x1y1x2y2
        bbox = np.array(bbox, dtype=int)
        
        queryimg = cv2.imread(os.path.join(self.queryimg_dir, queryimg_name))
        queryimg = cv2.cvtColor(queryimg, cv2.COLOR_BGR2RGB)
        
        rsimg = cv2.imread(os.path.join(self.rsimg_dir, rsimg_name))
        # my_rsimg = visualize_bbox(rsimg, bbox, cls_name)
        # cv2.imshow('img', my_rsimg)
        # cv2.waitKey(0)
        rsimg = cv2.cvtColor(rsimg, cv2.COLOR_BGR2RGB)
        if self.augment:
            rs_transformed = self.rs_transform(image=rsimg, bboxes=[list(bbox)+[cls_name]])
            rsimg = rs_transformed['image']
            bbox = rs_transformed['bboxes'][0][0:4]

        # Norm, to tensor
        if self.transform is not None:
            rsimg = self.transform(rsimg.copy())
            queryimg = self.transform(queryimg.copy())
        
        query_featuremap_hw = self.query_featuremap_hw
        click_hw = (int(click_xy[1]), int(click_xy[0]))
        
        mat_clickhw = np.zeros((query_featuremap_hw[0], query_featuremap_hw[1]), dtype=np.float32)
        click_h = [pow(one-click_hw[0],2) for one in range(query_featuremap_hw[0])]
        click_w = [pow(one-click_hw[1],2) for one in range(query_featuremap_hw[1])]
        norm_hw = pow(query_featuremap_hw[0]*query_featuremap_hw[0] + query_featuremap_hw[1]*query_featuremap_hw[1], 0.5)
        for i in range(query_featuremap_hw[0]):
            for j in range(query_featuremap_hw[1]):
                tmp_val = 1 - (pow(click_h[i]+click_w[j], 0.5)/norm_hw)
                mat_clickhw[i, j] = tmp_val * tmp_val

        query_feat_embdding = self.posemb_sincos_2d(8, 8, 512)
        reference_feat_embdding = self.posemb_sincos_2d(64, 64, 512)

        
        # return queryimg, rsimg, mat_clickhw,  np.array(bbox, dtype=np.float32), idx, queryimg_name, rsimg_name
        return queryimg, rsimg, mat_clickhw,  np.array(bbox, dtype=np.float32), idx

    
if __name__=='__main__':
    import pickle
    parser = argparse.ArgumentParser(
        description='cross-view object geo-localization')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--num_workers', default=24, type=int, help='num workers for data loading')

    parser.add_argument('--max_epoch', default=25, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--emb_size', default=512, type=int, help='embedding dimensions')
    parser.add_argument('--img_size', default=1024, type=int, help='image size')
    parser.add_argument('--data_root', type=str, default="/home/zzh/Desktop/CVOGL", help='path to the root folder of all dataset')
    parser.add_argument('--data_name', default='CVOGL_DroneAerial', type=str, help='CVOGL_DroneAerial/CVOGL_SVI')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH')
    parser.add_argument('--print_freq', '-p', default=50, type=int, metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--savename', default='default', type=str, help='Name head for saved model')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--beta', default=1.0, type=float, help='the weight of cls loss')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='test')
    parser.add_argument('--val', dest='val', default=False, action='store_true', help='val')
    
    global args, anchors_full
    args = parser.parse_args()

    input_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RSDataset(data_root=args.data_root,
                         data_name=args.data_name,
                         split_name='train',
                         img_size=args.img_size,
                         transform=input_transform,
                         augment=True)
    train_dataset1 = RSDataset(data_root=args.data_root,
                         data_name=args.data_name,
                         split_name='train',
                         img_size=args.img_size,
                         transform=input_transform,
                         augment=True)
    val_dataset = RSDataset(data_root=args.data_root,
                         data_name=args.data_name,
                         split_name='val',
                         img_size = args.img_size,
                         transform=input_transform)
    with open('/home/zzh/Desktop/CVOGL/CVOGL_DroneAerial/similarity_DroneAerial_rsimg_SSIM_train.pkl', "rb") as f:
            sim_dict = pickle.load(f)
    # train_dataset.shuffle(sim_dict=sim_dict)
    
    train_loader = DataLoader(train_dataset1, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, drop_last=False, num_workers=args.num_workers)
    idxlist = train_loader.dataset.samples
    
    # for batch_idx, (query_imgs, rs_imgs, mat_clickxy, ori_gt_bbox, _) in enumerate(train_loader):
    #     batch_indices = train_loader.dataset.indices[batch_idx * train_loader.batch_size : (batch_idx + 1) * train_loader.batch_size]
    #     print(f'Batch index: {batch_idx}, Sample indices: {batch_indices}')
    # for batch_idx, (query_imgs, rs_imgs, mat_clickxy, ori_gt_bbox, _) in enumerate(train_loader):
    #     print(len(query_imgs))
    bar = tqdm(train_loader, total=len(train_loader))
    for query_imgs, rs_imgs, mat_clickxy, ori_gt_bbox, _ in bar:
        print(len(query_imgs))