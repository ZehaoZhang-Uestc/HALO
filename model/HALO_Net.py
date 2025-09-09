# -*- coding:utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .darknet import *
import torchvision.models as models
from .LskNet_backbone import LSKNet
from einops.einops import rearrange
from .transformer import LocalFeatureTransformer, LoFTREncoderLayer
from .FSRA import build_cluster_feature
import time
from losses.triplet_loss import Tripletloss,TripletLoss
from losses.cal_loss import cal_triplet_loss
from mmengine.visualization import Visualizer


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model=256, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]



class InfoNCE(nn.Module):

    def __init__(self, embed_dim=512, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.projector = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(embed_dim * 4 * 4, 10))
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.device = device

    def forward(self, featA, featB):
        B, C, _, _ = featA.shape
        featA = self.projector(featA)
        featB = self.projector(featB)

        featA = F.normalize(featA, dim=-1)
        # print('featA.size',featA.size()) 4*10
        featB = F.normalize(featB, dim=-1)

        logits_C = self.logit_scale * featA @ featB.T
        logits_D = logits_C.T

        logits_A = self.logit_scale * featA @ featA.T  #self attention
        logits_B = self.logit_scale * featB @ featB.T

        labels = torch.arange(len(logits_A), dtype=torch.long, device=self.device)
        loss_self = (self.loss_function(logits_A, labels) + self.loss_function(logits_B, labels)) / 2

        loss_mutual = (self.loss_function(logits_C, labels) + self.loss_function(logits_D, labels)) / 2
        loss = loss_mutual
        # loss = loss_mutual + loss_self

        return loss*0

class InfoNCE_sem(nn.Module):

    def __init__(self, embed_dim=512, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.device = device

    def forward(self, featA, featB, weights):
       
        featA = F.normalize(featA, dim=-1)
        # print('featA.size',featA.size()) 4*10
        featB = F.normalize(featB, dim=-1)

        logits_C = self.logit_scale * featA @ featB.T
        logits_D = logits_C.T

        logits_A = self.logit_scale * featA @ featA.T  #self attention
        logits_B = self.logit_scale * featB @ featB.T

        labels = torch.arange(len(logits_A), dtype=torch.long, device=self.device)
        loss_self = (self.loss_function(logits_A, labels) + self.loss_function(logits_B, labels)) / 2

        loss_mutual = (self.loss_function(logits_C, labels) + self.loss_function(logits_D, labels)) / 2
        loss = loss_mutual
        # loss = loss_mutual + loss_self

        return loss * weights
    

class InfoNCE_sem_serial(nn.Module):

    def __init__(self, embed_dim=512, block=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.projector = nn.Linear(embed_dim * block, 512)
        self.device = device
 

    def forward(self, featA, featB):
    
        featA = torch.cat(featA, dim=1)
        featB = torch.cat(featB, dim=1)
     
        featA = self.projector(featA)
        featB = self.projector(featB)
        featA = F.normalize(featA, dim=-1)
        featB = F.normalize(featB, dim=-1)

        logits_C = self.logit_scale * featA @ featB.T
        logits_D = logits_C.T

        logits_A = self.logit_scale * featA @ featA.T  #self attention
        logits_B = self.logit_scale * featB @ featB.T

        labels = torch.arange(len(logits_A), dtype=torch.long, device=self.device)
        loss_self = (self.loss_function(logits_A, labels) + self.loss_function(logits_B, labels)) / 2

        loss_mutual = (self.loss_function(logits_C, labels) + self.loss_function(logits_D, labels)) / 2
        loss = loss_mutual
        # loss = loss_mutual + loss_self
        if self.training:
            return loss    
        else:
            return featA, featB
        


class MyResnet(nn.Module):
    def __init__(self):
        super(MyResnet, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.avgpool = nn.Sequential()
        self.base_model.fc = nn.Sequential()

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x1 = self.base_model.layer1(x)
        #print(('x1', x.shape), flush=True)
        x2 = self.base_model.layer2(x1)
        #print(('x2', x.shape), flush=True)
        x3 = self.base_model.layer3(x2)
        #print(('x3', x.shape), flush=True)
        x4 = self.base_model.layer4(x3)
        #print(('x4', x.shape), flush=True)
        return x2, x3, x4

class FPN(nn.Module):
    def __init__(self, in_channels_p2, in_channels_p3, in_channels_p4, out_channels=512):
        super(FPN, self).__init__()
        # 1x1 卷积以调整通道数
        self.conv_p2 = nn.Conv2d(in_channels_p2, out_channels, kernel_size=1)
        self.conv_p3 = nn.Conv2d(in_channels_p3, out_channels, kernel_size=1)
        self.conv_p4 = nn.Conv2d(in_channels_p4, out_channels, kernel_size=1)
        self.downsample_p2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, p2, p3, p4):
        # 调整通道数
        p2 = self.conv_p2(p2)
        p3 = self.conv_p3(p3)
        p4 = self.conv_p4(p4)

        # 获取 P3 的尺寸
        p3_size = p3.size()[2:]

        # 上采样 P4 到 P3 的尺寸
        p4_upsampled = F.interpolate(p4, size=p3_size, mode='bilinear', align_corners=False)
        
        # 融合 P3 和上采样后的 P4
        p3_fused = p3 + p4_upsampled

        # 下采样 P2 到 P3 的尺寸
        p2_downsampled = self.downsample_p2(p2)
        
        # 融合 P3_fused 和下采样后的 P2
        p3_final = p3_fused + p2_downsampled

        return p3_final


   

class CrossViewFusionModule(nn.Module):
    def __init__(self):
        super(CrossViewFusionModule, self).__init__()

    # normlized global_query:B, D
    # normlized value: B, D, H, W
    def forward(self, global_query, value):
        global_query = F.normalize(global_query, p=2, dim=-1)
        value = F.normalize(value, p=2, dim=1)

        B, D, W, H = value.shape
        new_value = value.permute(0, 2, 3, 1).view(B, W*H, D)
        score = torch.bmm(global_query.view(B, 1, D), new_value.transpose(1,2))
        score = score.view(B, W*H)
        with torch.no_grad():
            score_np = score.clone().detach().cpu().numpy()
            max_score, min_score = score_np.max(axis=1), score_np.min(axis=1)
        
        attn = Variable(torch.zeros(B, H*W).cuda())
        for ii in range(B):
            attn[ii, :] = (score[ii] - min_score[ii]) / (max_score[ii] - min_score[ii])
        
        attn = attn.view(B, 1, W, H)
        context = attn * value
        return context, attn

class HALO(nn.Module):
    def __init__(self, emb_size=512, leaky=True):
        super(HALO, self).__init__()
        # Visual model
        self.query_resnet = MyResnet()

        # self.query_lsknet = LSKNet(embed_dims=[64, 128, 320, 512], drop_rate=0.1, drop_path_rate=0.1, depths=[2,2,4,2])
        # self.query_lsknet.load_state_dict(torch.load('./saved_models/lsk_s_backbone-e9d2e551.pth')['state_dict'],
        #                       strict=False)
        # print('yes_lsk_uav')


        # self.reference_darknet = Darknet(config_path='./model/yolov3_rs.cfg')
        # self.reference_darknet.load_weights('./saved_models/yolov3.weights')

        self.reference_lsknet = LSKNet(embed_dims=[64, 128, 320, 512], drop_rate=0.1, drop_path_rate=0.1, depths=[2,2,4,2])
        self.reference_lsknet.load_state_dict(torch.load('./saved_models/lsk_s_backbone-e9d2e551.pth')['state_dict'],
                              strict=False)
        print('yes_lsk_sat')
        use_instnorm=False
        
        self.combine_clickptns_conv = ConvBatchNormReLU(4, 3, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        
        self.fpn_query=FPN(128, 256, 512, emb_size)
        self.fpn_sat=FPN(128, 320, 512, emb_size)
        print('fpn_query')
        print('fpn_sat')
        

        self.feat_trans = LocalFeatureTransformer()

        self.feature_cluster = build_cluster_feature(block=3)#3
        print('block=3')

        self.crossview_fusionmodule_1 = CrossViewFusionModule()
        # self.crossview_fusionmodule_2 = CrossViewFusionModule()
        print('CrossViewFusionModule(old_glable)')

        # self.crossview_fusionmodule = Multi_CrossViewFusionModule(dim=512)
        # print('Multi_CrossViewFusionModule_Igarss')
		
        self.sat_selfattn_l1 = LoFTREncoderLayer(emb_size, 8, 'linear')
        self.sat_selfattn_l2 = LoFTREncoderLayer(emb_size, 8, 'linear')

        # planes = emb_size
        # self.local = nn.Sequential(
        #     nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(planes),
        #     )
        
        # self.qry_local = nn.Sequential(
        #     nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(planes),
        #     )

        self.query_visudim = emb_size
        self.reference_visudim = emb_size

        self.query_mapping_visu_l1 = ConvBatchNormReLU(self.query_visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        # self.reference_mapping_visu_l1 = ConvBatchNormReLU(self.reference_visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)

        # self.query_mapping_visu_l2 = ConvBatchNormReLU(self.query_visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.reference_mapping_visu_l2 = ConvBatchNormReLU(self.reference_visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)

        self.query_mapping_visu = ConvBatchNormReLU(self.query_visudim, emb_size, 3, 2, 1, 1, leaky=leaky, instance=use_instnorm)
        self.reference_mapping_visu = ConvBatchNormReLU(self.reference_visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.pos_encoding = PositionEncodingSine(d_model=512)
        self.InfoNCE_sem_serial = InfoNCE_sem_serial(embed_dim=emb_size, block=3)
        # self.InfoNCE_sem = InfoNCE_sem(embed_dim=emb_size)
        # self.weights_infonce = torch.tensor([1., 0.5, 0.1], dtype=torch.float32)

        # layer_scale_init_value = 1e-2
        # self.layer_scale_1 = nn.Parameter(
        #     layer_scale_init_value * torch.ones((emb_size)), requires_grad=True)
        # self.layer_scale_2 = nn.Parameter(
        #     layer_scale_init_value * torch.ones((emb_size)), requires_grad=True)
        ## output head
        self.fcn_out = torch.nn.Sequential(
                ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm),
                nn.Conv2d(emb_size//2, 9*5, kernel_size=1))
        
    def cal_trip_loss(self, qry_cluster_feat, sat_cluster_feat):
        pic_number = qry_cluster_feat[0].size()[0]
        # print(pic_number)
        labels = torch.arange(0, pic_number)
        labels = labels.cuda().detach()
        triplet_loss = Tripletloss(margin=0.3)
        f_triplet_loss = cal_triplet_loss(qry_cluster_feat,sat_cluster_feat,labels,triplet_loss)
        return f_triplet_loss

    def forward(self, query_imgs, reference_imgs, mat_clickptns):
        mat_clickptns = mat_clickptns.unsqueeze(1)
        
        query_imgs = self.combine_clickptns_conv(torch.cat((query_imgs, mat_clickptns), dim=1) )

        query_fvisu_1,query_fvisu_2,query_fvisu_3 = self.query_resnet(query_imgs)

        ########### 无人机视角
        query_fvisu = self.fpn_query(query_fvisu_1,query_fvisu_2,query_fvisu_3) #FPN
        original_qry_shape = query_fvisu.size()
        ########### 卫星视角
        reference_raw_fvisu = self.reference_lsknet(reference_imgs)
        # reference_fvisu = reference_raw_fvisu[3]
        reference_fvisu = self.fpn_sat(reference_raw_fvisu[1],reference_raw_fvisu[2],reference_raw_fvisu[3]) #FPN
        original_sat_shape = reference_fvisu.size()
        B, D, Hreference, Wreference = reference_fvisu.shape
        B, D, Hquery, Wquery = query_fvisu.shape
        # print(query_fvisu.size())
        # print(reference_fvisu.size())

        ############新增融合
        # 无人机到卫星空间
        feat_qry_1 = rearrange(self.pos_encoding(query_fvisu), 'n c h w -> n (h w) c')
        feat_sat_1 = rearrange(self.pos_encoding(reference_fvisu), 'n c h w -> n (h w) c')
        cross_attn_qryfeat_ = self.sat_selfattn_l1(feat_qry_1, feat_sat_1)
        cross_attn_qryfeat = rearrange(cross_attn_qryfeat_, 'n (h w) c -> n c h w ', h=Hquery, w=Wquery)

        norm_qryfeat_1 = self.query_mapping_visu_l1(cross_attn_qryfeat)
  
        query_gvisu_l1 = torch.mean(norm_qryfeat_1.view(B, D, Hquery*Wquery), dim=2, keepdims=False).view(B, D)
       

        # 卫星到无人机空间
        feat_qry_2 = rearrange(self.pos_encoding(query_fvisu), 'n c h w -> n (h w) c')
        feat_sat_2 = rearrange(self.pos_encoding(reference_fvisu), 'n c h w -> n (h w) c')
        cross_attn_satfeat_ = self.sat_selfattn_l2(feat_sat_2, feat_qry_2)
        cross_attn_satfeat = rearrange(cross_attn_satfeat_, 'n (h w) c -> n c h w ', h=Hreference, w=Wreference)

       
        norm_satfeat_2 = self.reference_mapping_visu_l2(cross_attn_satfeat)
        
        fused_features, attn_score_1 = self.crossview_fusionmodule_1(query_gvisu_l1, norm_satfeat_2)


        attn_score = attn_score_1.squeeze(1)

        outbox = self.fcn_out(fused_features)

        if self.training:
            # time1 = time.time()
            feat_qry_cluster = self.feature_cluster(cross_attn_qryfeat_)
            feat_sat_cluster = self.feature_cluster(cross_attn_satfeat_)
            # time2 = time.time()

            infonce_loss = self.InfoNCE_sem_serial(feat_qry_cluster, feat_sat_cluster)

            # infonce_loss = torch.tensor(0.0, device=fused_features.device)

            # for i in range(len(feat_qry_cluster)):
            #     infonce_loss += self.InfoNCE_sem(feat_qry_cluster[i], feat_sat_cluster[i], self.weights_infonce[i])

            return outbox, attn_score, infonce_loss
        
        else:
            return outbox, attn_score
    
    def get_features(self, query_imgs, reference_imgs, mat_clickptns):
        mat_clickptns = mat_clickptns.unsqueeze(1)
        
        query_imgs = self.combine_clickptns_conv(torch.cat((query_imgs, mat_clickptns), dim=1))

        # query_raw_fvisu = self.query_lsknet(query_imgs)
        # llev_uavfeats, hlev_uavfeats = query_raw_fvisu[0], query_raw_fvisu[3]

        llev_uavfeats, hlev_uavfeats = self.query_resnet(query_imgs)
        
        reference_raw_fvisu = self.reference_lsknet(reference_imgs)
        llev_satfeats, hlev_satfeats = reference_raw_fvisu[0], reference_raw_fvisu[3]

        return hlev_uavfeats, hlev_satfeats
    
    def get_cluster_features(self, query_imgs, reference_imgs, mat_clickptns):
        
        mat_clickptns = mat_clickptns.unsqueeze(1)
        
        query_imgs = self.combine_clickptns_conv(torch.cat((query_imgs, mat_clickptns), dim=1) )

        query_fvisu_1,query_fvisu_2,query_fvisu_3 = self.query_resnet(query_imgs)

        ########### 无人机视角
        query_fvisu = self.fpn_query(query_fvisu_1,query_fvisu_2,query_fvisu_3) #FPN
        original_qry_shape = query_fvisu.size()
        ########### 卫星视角
        reference_raw_fvisu = self.reference_lsknet(reference_imgs)
        # reference_fvisu = reference_raw_fvisu[3]
        reference_fvisu = self.fpn_sat(reference_raw_fvisu[1],reference_raw_fvisu[2],reference_raw_fvisu[3]) #FPN
        original_sat_shape = reference_fvisu.size()
        B, D, Hreference, Wreference = reference_fvisu.shape
        B, D, Hquery, Wquery = query_fvisu.shape
        # print(query_fvisu.size())
        # print(reference_fvisu.size())

        ############新增融合
        # 无人机到卫星空间
        feat_qry_1 = rearrange(self.pos_encoding(query_fvisu), 'n c h w -> n (h w) c')
        feat_sat_1 = rearrange(self.pos_encoding(reference_fvisu), 'n c h w -> n (h w) c')
        cross_attn_qryfeat_ = self.sat_selfattn_l1(feat_qry_1, feat_sat_1)

        feat_qry_2 = rearrange(self.pos_encoding(query_fvisu), 'n c h w -> n (h w) c')
        feat_sat_2 = rearrange(self.pos_encoding(reference_fvisu), 'n c h w -> n (h w) c')
        cross_attn_satfeat_ = self.sat_selfattn_l2(feat_sat_2, feat_qry_2)
        if torch.isnan(cross_attn_qryfeat_).any():
                nan_count = torch.sum(torch.isnan(cross_attn_qryfeat_)).item()
                print(f"Number of NaN values cross_attn_qryfeat_: {nan_count}")
                # x_sort[i] = torch.nan_to_num(x_sort[i], nan=0.0)
                # print("NaN detected in tensor at inference stage:", 'x_sort[i]')
        if torch.isnan(cross_attn_satfeat_).any():
                nan_count = torch.sum(torch.isnan(cross_attn_satfeat_)).item()
                print(f"Number of NaN values cross_attn_satfeat_: {nan_count}")
                # x_sort[i] = torch.nan_to_num(x_sort[i], nan=0.0)
                # print("NaN detected in tensor at inference stage:", 'x_sort[i]')

        # print(cross_attn_qryfeat_.size())
        feat_qry_cluster = self.feature_cluster(cross_attn_qryfeat_)
        
        feat_sat_cluster = self.feature_cluster(cross_attn_satfeat_)
        # print(len(feat_qry_cluster))
        # print(feat_qry_cluster[0].size())
        
        feat_qry_serial, feat_sat_serial = self.InfoNCE_sem_serial(feat_qry_cluster, feat_sat_cluster)

        return feat_qry_serial, feat_sat_serial

    
    


