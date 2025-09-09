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


def calculate_nearest(query_features, reference_features, neighbour_range=64, step_size=1000):

    
    Q = len(query_features)
    
    steps = Q // step_size + 1
    
    similarity_query = []
    similarity_reference = []
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = min(start + step_size, Q)
          
        sim_tmp_query = query_features[start:end] @ query_features.T

        sim_tmp_reference = reference_features[start:end] @ reference_features.T

        sim_tmp = sim_tmp_reference + sim_tmp_query
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    nearest_dict = get_top_k_similar_ids(similarity, neighbour_range)

    return nearest_dict

def calculate_nearest_mutual(query_features, reference_features, neighbour_range=64, step_size=1000):

    
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

def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    uav_features_list = []
    sat_features_list = []
    # device='cuda' if torch.cuda.is_available() else 'cpu'
  
    with torch.no_grad():
        
        for query_imgs, rs_imgs, mat_clickxy, ori_gt_bbox, _  in bar:
        
            with autocast():
                query_imgs, rs_imgs = query_imgs.cuda(), rs_imgs.cuda()
                mat_clickxy = mat_clickxy.cuda()
                uav_feats, sat_feats = model.module.get_features(query_imgs, rs_imgs, mat_clickxy)

                pooled_a = F.adaptive_avg_pool2d(uav_feats, (4, 4))
                pooled_b = F.adaptive_avg_pool2d(sat_feats, (4, 4))
                uav_feats = pooled_a.view(pooled_a.size(0), -1)
                sat_feats = pooled_b.view(pooled_b.size(0), -1)

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    uav_feats = F.normalize(uav_feats, dim=-1)
                    sat_feats = F.normalize(sat_feats, dim=-1)

            
            # save features in fp32 for sim calculation
            uav_features_list.append(uav_feats.to(torch.float32))
            sat_features_list.append(sat_feats.to(torch.float32))
      
        # keep Features on GPU
        img_uav_features = torch.cat(uav_features_list, dim=0)
        print('总计特征图大小：',img_uav_features.size) 
        img_sat_features = torch.cat(sat_features_list, dim=0) 
        
        
    if train_config.verbose:
        bar.close()
        
    return img_uav_features, img_sat_features


def calc_sim(config, model, sample_dataloader, step_size=500, cleanup=True):
    
    
    print("\nExtract Features:")
    query_features, reference_features = predict(config, model, sample_dataloader) 
    
    near_dict = calculate_nearest_mutual(query_features=query_features,
                                  reference_features=reference_features,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)
            
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features,  query_features
        gc.collect()
        
    return  near_dict

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