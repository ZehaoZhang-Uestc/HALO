import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class ClassBlock(nn.Module):
    def __init__(self, input_dim, droprate, relu=True, bnorm=False, num_bottleneck=256, linear=True):
        super(ClassBlock, self).__init__()
        
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)


        self.add_block = add_block
      
    def forward(self, x):
        # print('x',x.size())
        # x = self.add_block(x)#zzh 去掉addblock
        if self.training:
            return x        
        else:
            return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)




class build_cluster_feature(nn.Module):
    def __init__(self, block = 3):
        super(build_cluster_feature, self).__init__()
        self.in_planes = 256
        self.classifier1 = ClassBlock(self.in_planes,0.5)
        self.block = block
        for i in range(self.block):
            name = 'classifier_heat' + str(i+1)
            setattr(self, name, ClassBlock(self.in_planes, 0.5))


    def forward(self, x):
     
        part_features = x

        if self.block==1:
            return part_features
        
        # heat_result = self.get_heartmap_pool(part_features)
        heat_result, cluster_labels = self.get_heartmap_pool_cluster(part_features)
       
        # print("heat_result",heat_result.size())
        y = self.part_classifier(self.block, heat_result, cls_name='classifier_heat')
        # print('y',len(y))
        
        if self.training:
            features = []
            for i in y:
                # print('i_type', type(i))
                # print('i_size', i.size())
                features.append(i)
            
            return features
        else:
            features = []
            for i in y:
                features.append(i)
           
            return features

    # batch batch batch batch batch batch batch batch batch batch batch batch   
    # def get_heartmap_pool_cluster(self, part_features, add_global=False, otherbranch=False):
    #     heatmap = torch.mean(part_features, dim=-1)
    #     size = part_features.size(1)
    #     arg = torch.argsort(heatmap, dim=1, descending=True)
    #     x_sort = [part_features[i, arg[i], :] for i in range(part_features.size(0))]
    #     x_sort = torch.stack(x_sort, dim=0)

    #     # 使用 K-means 进行聚类
    #     kmeans = KMeans(n_clusters=self.block)
    #     # cluster_labels = kmeans.fit_predict(x_sort.view(-1, x_sort.size(-1)).detach().cpu().numpy())
    #     cluster_labels = kmeans.fit_predict(torch.mean(x_sort.view(-1, x_sort.size(-1)),dim=-1,keepdim=True).detach().cpu().numpy())
    #     cluster_labels = torch.tensor(cluster_labels, device=x_sort.device).view(x_sort.size(0), -1)
    #     # 找到每个聚类的中心，并按大小排序
    #     centers = kmeans.cluster_centers_
    #     mean_centers = np.mean(centers, axis=1)
    #     sorted_indices = np.argsort(mean_centers)
    #     sorted_indices_desc = sorted_indices[::-1]
    #     label_mapping = {original_label: new_label for new_label, original_label in enumerate(sorted_indices_desc)}
    #     adjusted_labels = torch.tensor([label_mapping[label.item()] for label in cluster_labels.flatten()]).view(cluster_labels.size())

    #     # 处理聚类结果
    #     split_list = []
    #     for i in range(self.block):
    #         indices = (adjusted_labels == i).nonzero(as_tuple=True)[1]
    #         if indices.numel() > 0:
    #             split_list.append(x_sort[:, indices, :])

    #     # 计算每个聚类的均值
    #     split_means = [torch.mean(split, dim=1) for split in split_list if split.size(1) > 0]
    #     part_features_ = torch.stack(split_means, dim=2)

    #     if add_global:
    #         global_feat = torch.mean(part_features, dim=1).view(part_features.size(0), -1, 1).expand(-1, -1, self.block)
    #         part_features_ = part_features_ + global_feat
    #     if otherbranch:
    #         otherbranch_ = torch.mean(torch.stack(split_means[1:], dim=2), dim=-1)
    #         return part_features_, otherbranch_, adjusted_labels
    #     return part_features_, adjusted_labels
    # pic pic pic pic pic pic pic pic pic pic pic pic pic pic pic pic pic pic pic pic pic pic pic pic pic pic pic pic pic pic
    def get_heartmap_pool_cluster(self, part_features, add_global=False, otherbranch=False, visual_cluster=False, isquery=True, image_name=' '):
        heatmap = torch.mean(part_features, dim=-1)
        size = part_features.size(1)
        arg = torch.argsort(heatmap, dim=1, descending=True)
        x_sort = [part_features[i, arg[i], :] for i in range(part_features.size(0))]
        x_sort = torch.stack(x_sort, dim=0)
        # sort_heatmap = torch.mean(x_sort, dim=-1)
        # 使用 K-means 进行聚类
        part_features_=[]
        adjusted_labels_=[]
        for i in range(part_features.size(0)):
            kmeans = KMeans(n_clusters=self.block)
            # print('x_sort.view(-1, x_sort.size(-1))',x_sort[i].view(-1, x_sort[i].size(-1)).size())
            if torch.isnan(x_sort[i]).any():
                nan_count = torch.sum(torch.isnan(x_sort[i])).item()
                print(f"Number of NaN values: {nan_count}")
                x_sort[i] = torch.nan_to_num(x_sort[i], nan=0.0)
                print("NaN detected in tensor at inference stage:", 'x_sort[i]')
            if torch.isnan(torch.mean(x_sort[i].view(-1, x_sort[i].size(-1)),dim=-1,keepdim=True)).any():
                print("NaN detected in tensor at inference stage:", 'torch.mean(x_sort[i].view(-1, x_sort[i].size(-1)),dim=-1,keepdim=True)')

            cluster_labels = kmeans.fit_predict(torch.mean(x_sort[i].view(-1, x_sort[i].size(-1)),dim=-1,keepdim=True).detach().cpu().numpy())
            cluster_labels = torch.tensor(cluster_labels, device=x_sort[i].device).view(x_sort[i].size(0), -1)
            # 找到每个聚类的中心，并按大小排序
            centers = kmeans.cluster_centers_
            # print(centers.shape)
            mean_centers = np.mean(centers, axis=1)
            # print(mean_centers)
            sorted_indices = np.argsort(mean_centers)
            sorted_indices_desc = sorted_indices[::-1]
            
            label_mapping = {original_label: new_label for new_label, original_label in enumerate(sorted_indices_desc)}
            adjusted_labels = torch.tensor([label_mapping[label.item()] for label in cluster_labels.flatten()]).view(cluster_labels.size())

            if visual_cluster:
                if isquery:
                    original_images = "/home/zzh/Desktop/CVOGL/CVOGL_DroneAerial/query/"+image_name
                    output_path = '/home/zzh/Desktop/TGRS_PROJECT/DetGeo_bbox/Visual/B_cluster_results/query/'+image_name
                    self.save_cluster_masks(adjusted_labels, arg, original_images, output_path)
                else:
                    original_images = "/home/zzh/Desktop/CVOGL/CVOGL_DroneAerial/satellite/"+image_name
                    output_path = '/home/zzh/Desktop/TGRS_PROJECT/DetGeo_bbox/Visual/B_cluster_results/reference/'+image_name
                    self.save_cluster_masks(adjusted_labels, arg, original_images, output_path)

            # 处理聚类结果
            split_list = []
            for k in range(self.block):
                indices = (adjusted_labels.flatten() == k).nonzero(as_tuple=True)[0]
                # print(indices)
                if indices.numel() > 0:
                    split_list.append(x_sort[i][indices, :])

            # 计算每个聚类的均值
            split_means = [torch.mean(split, dim=0) for split in split_list if split.size(1) > 0]
            # print(split_means[0].size())
            part_features_temp = torch.stack(split_means, dim=1)
            # print('part_features_temp', part_features_temp.size())
            # print('adjusted_labels', adjusted_labels.size())
            part_features_.append(part_features_temp)
            adjusted_labels_.append(adjusted_labels)

        part_features_ = torch.stack(part_features_, dim=0)
        # print('part_features_', part_features_.size())
        adjusted_labels_ = torch.stack(adjusted_labels_, dim=0)
        # print('adjusted_labels_', adjusted_labels_.size())

        # if add_global:
        #     global_feat = torch.mean(part_features, dim=1).view(part_features.size(0), -1, 1).expand(-1, -1, self.block)
        #     part_features_ = part_features_ + global_feat
        # if otherbranch:
        #     otherbranch_ = torch.mean(torch.stack(split_means[1:], dim=2), dim=-1)
        #     return part_features_, otherbranch_, adjusted_labels
        return part_features_, adjusted_labels
    '''
    def get_heartmap_pool_cluster(self, part_features, add_global=False, otherbranch=False, visual_cluster=False, isquery=True, image_name=' '):
        heatmap = torch.mean(part_features, dim=-1)
        size = part_features.size(1)
        arg = torch.argsort(heatmap, dim=1, descending=True)
        x_sort = [part_features[i, arg[i], :] for i in range(part_features.size(0))]
        x_sort = torch.stack(x_sort, dim=0)
        # sort_heatmap = torch.mean(x_sort, dim=-1)
        # 使用 K-means 进行聚类
        part_features_=[]
        adjusted_labels_=[]
        for i in range(part_features.size(0)):
            kmeans = KMeans(n_clusters=self.block)
            # print('x_sort.view(-1, x_sort.size(-1))',x_sort[i].view(-1, x_sort[i].size(-1)).size())
            cluster_labels = kmeans.fit_predict(x_sort[i].view(-1, x_sort[i].size(-1)).detach().cpu().numpy())
            cluster_labels = torch.tensor(cluster_labels, device=x_sort[i].device).view(x_sort[i].size(0), -1)
            # 找到每个聚类的中心，并按大小排序
            centers = kmeans.cluster_centers_
            # print(centers.shape)
            mean_centers = np.mean(centers, axis=1)
            # print(mean_centers)
            sorted_indices = np.argsort(mean_centers)
            sorted_indices_desc = sorted_indices[::-1]
            
            label_mapping = {original_label: new_label for new_label, original_label in enumerate(sorted_indices_desc)}
            adjusted_labels = torch.tensor([label_mapping[label.item()] for label in cluster_labels.flatten()]).view(cluster_labels.size())

            if visual_cluster:
                if isquery:
                    original_images = "/home/zzh/Desktop/CVOGL/CVOGL_DroneAerial/query/"+image_name
                    output_path = '/home/zzh/Desktop/TGRS_PROJECT/DetGeo_bbox/Visual/B_cluster_results/query/'+image_name
                    self.save_cluster_masks(adjusted_labels, arg, original_images, output_path)
                else:
                    original_images = "/home/zzh/Desktop/CVOGL/CVOGL_DroneAerial/satellite/"+image_name
                    output_path = '/home/zzh/Desktop/TGRS_PROJECT/DetGeo_bbox/Visual/B_cluster_results/reference/'+image_name
                    self.save_cluster_masks(adjusted_labels, arg, original_images, output_path)

            # 处理聚类结果
            split_list = []
            for k in range(self.block):
                indices = (adjusted_labels == k).nonzero(as_tuple=True)[1]
                if indices.numel() > 0:
                    split_list.append(x_sort[i][indices, :])

            # 计算每个聚类的均值
            split_means = [torch.mean(split, dim=0) for split in split_list if split.size(1) > 0]
            # print(split_means[0].size())
            part_features_temp = torch.stack(split_means, dim=1)
            # print('part_features_temp', part_features_temp.size())
            # print('adjusted_labels', adjusted_labels.size())
            part_features_.append(part_features_temp)
            adjusted_labels_.append(adjusted_labels)

        part_features_ = torch.stack(part_features_, dim=0)
        # print('part_features_', part_features_.size())
        adjusted_labels_ = torch.stack(adjusted_labels_, dim=0)
        # print('adjusted_labels_', adjusted_labels_.size())

        # if add_global:
        #     global_feat = torch.mean(part_features, dim=1).view(part_features.size(0), -1, 1).expand(-1, -1, self.block)
        #     part_features_ = part_features_ + global_feat
        # if otherbranch:
        #     otherbranch_ = torch.mean(torch.stack(split_means[1:], dim=2), dim=-1)
        #     return part_features_, otherbranch_, adjusted_labels
        return part_features_, adjusted_labels
    '''
   
    def get_heartmap_pool(self, part_features, add_global=False, otherbranch=False):
        heatmap = torch.mean(part_features,dim=-1)
        size = part_features.size(1)
        arg = torch.argsort(heatmap, dim=1, descending=True)
        x_sort = [part_features[i, arg[i], :] for i in range(part_features.size(0))]
        x_sort = torch.stack(x_sort, dim=0)

        split_each = size / self.block
        split_list = [int(split_each) for i in range(self.block - 1)]
        split_list.append(size - sum(split_list))
        split_x = x_sort.split(split_list, dim=1)

        split_list = [torch.mean(split, dim=1) for split in split_x]
        part_featuers_ = torch.stack(split_list, dim=2)
        if add_global:
            global_feat = torch.mean(part_features, dim=1).view(part_features.size(0), -1, 1).expand(-1, -1, self.block)
            part_featuers_ = part_featuers_ + global_feat
        if otherbranch:
            otherbranch_ = torch.mean(torch.stack(split_list[1:], dim=2), dim=-1)
            return part_featuers_, otherbranch_
        return part_featuers_

    def part_classifier(self, block, x, cls_name='classifier_lpn'):
        part = {}
        predict = {}
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = cls_name + str(i+1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(block):
            y.append(predict[i])
       
        return y

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

if __name__ == '__main__':
    mx = build_cluster_feature()
    input_tensor = torch.rand(16, 32, 256)
    output = mx(input_tensor)
    print(len(output))
    print(output[0].size())