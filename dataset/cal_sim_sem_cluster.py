import gc
import torch
import time
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F

def get_top_k_similar_ids(similarity_matrix, neighbour_range=64):
    # 获取前 neighbour_range + 2 个相似度分数和对应的索引
    topk_scores, topk_ids = torch.topk(similarity_matrix, k=neighbour_range + 2, dim=1)

    sim_dict = {}
    num_queries = similarity_matrix.shape[0]
    
    for i in range(num_queries):
        # 取出前 neighbour_range + 1 个相似项，去掉自身（索引为 i 的项）
        sim_dict[i] = topk_ids[i][topk_ids[i] != i][:neighbour_range].tolist()
    
    return sim_dict


def calculate_nearest_self(query_features, reference_features, neighbour_range=64, step_size=1000):

    
    Q = len(query_features)
    
    steps = Q // step_size + 1
    
    similarity_query = []
    similarity_reference = []
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = min(start + step_size, Q)
          
        # sim_tmp_query = query_features[start:end] @ query_features.T

        # sim_tmp_reference = reference_features[start:end] @ reference_features.T

        sim_tmp_query_reference = query_features[start:end] @ reference_features.T

        sim_tmp_reference_query = reference_features[start:end] @ query_features.T

        sim_tmp = sim_tmp_query_reference + sim_tmp_reference_query
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    similarity_min = similarity.min()
    similarity_max = similarity.max()

    # 进行归一化
    similarity = (similarity - similarity_min) / (similarity_max - similarity_min)
    
    return similarity

    

def calculate_nearest_mutual(query_features, reference_features, neighbour_range=64, step_size=500):

    
    Q = len(query_features)
    
    steps = Q // step_size + 1
    
    similarity_query_reference = []
    similarity_reference_query = []
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = min(start + step_size, Q)
          
        sim_tmp_query_reference = query_features[start:end] @ reference_features.T

        sim_tmp_reference_query = reference_features[start:end] @ query_features.T

        
        sim_tmp = sim_tmp_query_reference + sim_tmp_reference_query
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    nearest_dict = get_top_k_similar_ids(similarity, neighbour_range)

    return nearest_dict

def calculate_nearest_self_mutual(query_features, reference_features, epoch, max_epoch, neighbour_range=64, step_size=1000):

    rate = epoch/max_epoch

    Q = len(query_features)
    
    steps = Q // step_size + 1
    
    similarity_query_reference = []
    similarity_reference_query = []
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = min(start + step_size, Q)
          
        sim_tmp_query_reference = query_features[start:end] @ reference_features.T

        sim_tmp_reference_query = reference_features[start:end] @ query_features.T

        sim_tmp_query = query_features[start:end] @ query_features.T

        sim_tmp_reference = reference_features[start:end] @ reference_features.T


        # sim_tmp = (1-rate)*(sim_tmp_query_reference + sim_tmp_reference_query)+rate*(sim_tmp_reference + sim_tmp_query)
        sim_tmp = 0.5 * (sim_tmp_query_reference + sim_tmp_reference_query) + 0.5 * (sim_tmp_reference + sim_tmp_query)
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    nearest_dict = get_top_k_similar_ids(similarity, neighbour_range)

    return nearest_dict

def predict_distribute(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    uav_clauster_feats_0 = []
    uav_clauster_feats_1 = []
    uav_clauster_feats_2 = []
    sat_clauster_feats_0 = []
    sat_clauster_feats_1 = []
    sat_clauster_feats_2 = []
    # device='cuda' if torch.cuda.is_available() else 'cpu'
  
    with torch.no_grad():
        
        for query_imgs, rs_imgs, mat_clickxy, ori_gt_bbox, _  in bar:
        
            with autocast():
                query_imgs, rs_imgs = query_imgs.cuda(), rs_imgs.cuda()
                mat_clickxy = mat_clickxy.cuda()
                uav_clauster_feats, sat_clauster_feats = model.module.get_cluster_features(query_imgs, rs_imgs, mat_clickxy)

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    uav_feats_0 = F.normalize(uav_clauster_feats[0], dim=-1)
                    uav_feats_1 = F.normalize(uav_clauster_feats[1], dim=-1)
                    uav_feats_2 = F.normalize(uav_clauster_feats[2], dim=-1)
                    sat_feats_0 = F.normalize(sat_clauster_feats[0], dim=-1)
                    sat_feats_1 = F.normalize(sat_clauster_feats[1], dim=-1)
                    sat_feats_2 = F.normalize(sat_clauster_feats[2], dim=-1)

            # save features in fp32 for sim calculation
            uav_clauster_feats_0.append(uav_feats_0.to(torch.float32))
            uav_clauster_feats_1.append(uav_feats_1.to(torch.float32))
            uav_clauster_feats_2.append(uav_feats_2.to(torch.float32))
            sat_clauster_feats_0.append(sat_feats_0.to(torch.float32))
            sat_clauster_feats_1.append(sat_feats_1.to(torch.float32))
            sat_clauster_feats_2.append(sat_feats_2.to(torch.float32))
      
        # keep Features on GPU
        img_uav_features_0 = torch.cat(uav_clauster_feats_0, dim=0)
        img_uav_features_1 = torch.cat(uav_clauster_feats_1, dim=0)
        img_uav_features_2 = torch.cat(uav_clauster_feats_2, dim=0)
    
        img_sat_features_0 = torch.cat(sat_clauster_feats_0, dim=0)
        img_sat_features_1 = torch.cat(sat_clauster_feats_0, dim=0)
        img_sat_features_2 = torch.cat(sat_clauster_feats_0, dim=0) 
   
    if train_config.verbose:
        bar.close()
        
    return img_uav_features_0,img_uav_features_1,img_uav_features_2,img_sat_features_0,img_sat_features_1,img_sat_features_2

def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    uav_clauster_feats = []
    sat_clauster_feats = []
    
    # device='cuda' if torch.cuda.is_available() else 'cpu'
  
    with torch.no_grad():
        
        for query_imgs, rs_imgs, mat_clickxy, ori_gt_bbox, _,  in bar:#zzh 看输入的datasets是5个（水平）还是6个（旋转）要改一下
        
            with autocast():
                query_imgs, rs_imgs = query_imgs.cuda(), rs_imgs.cuda()
                mat_clickxy = mat_clickxy.cuda()
                uav_clauster_feats_item, sat_clauster_feats_item = model.module.get_cluster_features(query_imgs, rs_imgs, mat_clickxy)

            # save features in fp32 for sim calculation
            uav_clauster_feats.append(uav_clauster_feats_item.to(torch.float32))
            sat_clauster_feats.append(sat_clauster_feats_item.to(torch.float32))
           
        # keep Features on GPU
        img_uav_features_0 = torch.cat(uav_clauster_feats, dim=0)
        img_sat_features_0 = torch.cat(sat_clauster_feats, dim=0)
        
    if train_config.verbose:
        bar.close()
        
    return img_uav_features_0,img_sat_features_0


def calc_sim(config, model, sample_dataloader, step_size=500, cleanup=True):
    
    
    print("\nExtract Features:")
    img_uav_features_0,img_sat_features_0 = predict(config, model, sample_dataloader) 
    
    similarity = calculate_nearest_self(img_uav_features_0,img_sat_features_0,step_size=step_size)
    
    nearest_dict = get_top_k_similar_ids(similarity, neighbour_range=32)
         
    # cleanup and free memory on GPU
    if cleanup:
        del img_uav_features_0,img_sat_features_0
        gc.collect()
        
    return  nearest_dict

def calc_sim_adapt(config, model, sample_dataloader, epoch, maxepoch, step_size=500, cleanup=True):
    
    
    print("\nExtract Features:")
    query_features, reference_features = predict(config, model, sample_dataloader) 
    
    near_dict = calculate_nearest_self_mutual(query_features=query_features,
                                                reference_features=reference_features,
                                                epoch = epoch,
                                                max_epoch = maxepoch, 
                                                neighbour_range=config.neighbour_range,
                                                step_size=step_size)
            
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features,  query_features
        gc.collect()
        
    return  near_dict