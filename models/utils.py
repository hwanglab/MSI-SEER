import numpy as np

import os
from sys import platform
import glob

import pandas as pd

from PIL import Image

class data_feature_mat(object):
    def __init__(self, filepath=None, sample_ids=None, tile_names=None, graphs = None, data_mat=None, Nis=None):
        self.filePath = filepath

        self.data_mat = data_mat
        self.data_mat_ptr = None

        self.data_graphs = graphs

        self.sample_ids = sample_ids
        self.sample_tile_names = tile_names

        self.Nis = Nis

        return None

def get_MSI_data_with_tileinfo(str_data_path):     
    fname_list = glob.glob(os.path.join(str_data_path, '*.npz'))

    if platform == "linux" or platform == "linux2":
        str_sep = "/"
    elif platform == "win32" or platform == "win64":
        str_sep = "\\"
            
    img_features = []
    img_names = []
    img_tilenames = []
    img_Nis = []
    for fname in fname_list:        
        str_tmp = fname.split(str_sep)
        key = str_tmp[-1].split('.')    

        img_tmp = np.load(fname, allow_pickle=True)

        img_features.append(img_tmp['Fea'])
        img_tilenames.append(img_tmp['Info'])

        img_Nis.append(np.shape(img_tmp['Fea'])[0])

        img_names.append(key[0])

    Nis = np.reshape(np.vstack(img_Nis), [-1])

    str_flie_lable = os.path.join(str_data_path, 'label_info.txt')
    if os.path.exists(str_flie_lable):
        label_info = pd.read_csv(str_flie_lable, delimiter='\t')
        label_info_id = list(label_info["pID"].apply(str))

        sel_idx = [label_info_id.index(item) for item in img_names]
        img_labels = [label_info.iloc[idx, 1] for idx in sel_idx]
    else:
        img_labels = None
        
    return img_names, img_tilenames, img_features, img_labels, Nis 


def get_label_vector(labels):
    str_sub = 'MSI'

    if isinstance(labels[0], str) == False:
        labels = [str(elm) for elm in labels]

    # MSI-H
    Y_vec = np.zeros((len(labels), 1), dtype=np.int32)
    sel_idx = [idx for idx, str in enumerate(labels, 0) if ((str_sub in str) or (str=='1'))]

    Y_vec[sel_idx] = 1

    # NA
    sel_idx = [idx for idx, str in enumerate(labels, 0) if (str=='NA')]
    Y_vec[sel_idx] = -1

    return Y_vec

def get_x_y_dx_dy(patch_name):    
    if '_dx' not in patch_name:
        patch_name_ = patch_name.split('.png')[0]
        
        pos_str = patch_name_.find('_x')    
        _, x, y = patch_name_[pos_str:].split('_')
        
        x = int(x.split('x')[-1])
        y = int(y.split('y')[-1])
        
        dx = 1024
        dy = 1024
    else:
        patch_name_ = patch_name.split('.npy')[0]
        
        pos_str = patch_name_.find('_x')    
        _, x, y, dx, dy = patch_name_[pos_str:].split('_')

        x = int(x.split('x')[-1])
        y = int(y.split('y')[-1])
        dx = int(dx.split('dx')[-1])
        dy = int(dy.split('dy')[-1])

    return x, y, dx, dy

def fill_heatmap_grid_upscale\
    (wsi_size, tile_size, thumbnail_size, patch_ids, patch_preds):
    wsi_width, wsi_height = wsi_size
    width, height = thumbnail_size  
        
    heatmap_grid_shape = (np.array((wsi_height, wsi_width)) / tile_size).astype(np.int32)
    scale_factor_w = (width/heatmap_grid_shape[1]).astype(np.int32)
    scale_factor_h = (height/heatmap_grid_shape[0]).astype(np.int32)
        
    heatmap_grid = np.zeros\
        ((heatmap_grid_shape[0]*scale_factor_h, heatmap_grid_shape[1]*scale_factor_w))
    
    _, _, dx, dy = get_x_y_dx_dy(str(patch_ids[0]))
    assert dx == dy, 'Tiles must be square'

    n = dx
    for id, pred in zip(patch_ids, patch_preds):
        x_, y_, dx, dy = get_x_y_dx_dy(str(id))
        assert dx == dy, 'Tiles must be square'
        assert n == dx, 'Tiles must all be of the same shape'

        x = int(x_ / n)
        y = int(y_ / n)

        idx_row = np.arange(x*scale_factor_w, np.minimum(heatmap_grid.shape[1], (x+1)*scale_factor_w))
        idx_col = np.arange(y*scale_factor_h, np.minimum(heatmap_grid.shape[0], (y+1)*scale_factor_h))
        
        if (len(idx_row) == 0) or (len(idx_col) == 0):
            continue
            
        mat_vals = pred*np.ones((scale_factor_h, scale_factor_w), dtype=np.float32)
        
        heatmap_grid[np.ix_(idx_col, idx_row)] = mat_vals               
    
    heatmap_image =  Image.fromarray(heatmap_grid)
    if (heatmap_grid.shape[0] != height) or (heatmap_grid.shape[1] != width):
        heatmap_image = heatmap_image.resize((width, height), resample=Image.NEAREST)              
        
    return heatmap_image

def seed_everything(seed: int):
    import random
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True