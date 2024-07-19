import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import copy
import argparse

from sklearn.metrics import roc_auc_score

#- input argument
parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', type=str, default='0')

parser.add_argument('--input_feature_ViT_pretrained', type=bool, default=False)
parser.add_argument('--flag_dropconnect', type=bool, default=False)

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--minimum_epoch', type=int, default=75)

parser.add_argument('--sub_Ni', type=int, default=300)
parser.add_argument('--sel_instances', type=str, default='sampling') # sampling

# Variational dropout
parser.add_argument('--nMCsamples', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.5)

# DGP + Random feature expansion (RF)
parser.add_argument('--n_layers', type=int, default=6)
parser.add_argument('--n_RFs', type=int, default=100)
parser.add_argument('--ker_type', type=str, default='arccosin')  # arccosin, rbf

# ensemble learning
parser.add_argument('--exp_ensemble', type=bool, default=True)
parser.add_argument('--str_ensemble', type=str, default='full_training_data')
parser.add_argument('--n_runs', type=int, default=10)

parser.add_argument('--lr_init', type=int, default=1e-3)

parser.add_argument('--nExp', type=int, default=0)

parser.add_argument('--iter_print', type=bool, default=True)
parser.add_argument('--flag_train_model', type=bool, default=True)
# parser.add_argument('--flag_train_model', type=bool, default=False)
parser.add_argument('--save_directory', type=str, default='./')

parser.add_argument('--flag_mean_function', type=bool, default=True)
parser.add_argument('--model_ref_path', type=str, default='./model_weights/prior_means_fromMSIDETECT/')

# setting = parser.parse_args()
setting, unknown = parser.parse_known_args()

#- select a GPU
os.environ["CUDA_VISIBLE_DEVICES"] = setting.gpu_id

from models.wsl_binary_classifier_fea_vi_agg_ensemble import wsl_classifier as dgp_rf_agg_ens
from models.wsl_binary_classifier_fea_vi_agg_ensemble import cal_unc_quntities

from models.utils import get_MSI_data_with_tileinfo, data_feature_mat, get_label_vector

#- fix the random seed
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

n_selected_model = 0

log_np = lambda x: np.log(x + 1e-16)
sigmoid_np = lambda x: np.divide(1.0, 1.0 + np.exp(-2.0*x))


if __name__ == "__main__":
    #- data selection:
    N_MSIDETECT_models = 9    
    # N_MSIDETECT_models = 2
    
    if setting.nExp == 0:
        # Severance = Yonsei, STMary = STMary-Colon
        data_set_full = ['TCGA_CRC_Kather', 'Severance',  \
                         'Severance_new', 'Severance_2', 'STMary', 'CPATC_COAD', 'Mayo_Colon']  # 'TCGA_CRC', 
                
        str_trn_data = 'Severance'

    elif setting.nExp == 1:
        data_set_full = ['STMary_GC', 'GC_ICI', 'Molecular-Subtypes']
        
        str_trn_data = 'Pooled_STAD_wo_Yonsei_immuno'        
            
    #- experiment settings:
    print('Training = %s' % str_trn_data)

    # use the same number of GP layers
    setting.n_layers = 6    
    setting.save_directory = './results/save_models'
    
    DATA_PNG_PATH_PREFIX_MSIDETECT = "Z:\\PUBLIC\\lab_members\\sunho_park\\data\\WSIs\\WSI_features\\MSIDETECT_MODELS\\"
    RESULT_SAVE_PATH = "./results/"    

    # Training MSI   
    if setting.flag_train_model:     
        for ith_run in range(N_MSIDETECT_models):
            print('%dth model' % ith_run)

            str_model = 'MODEL_' + str(ith_run)   
                     
            img_names, img_tile_info, data_X, labels, Nis = \
                    get_MSI_data_with_tileinfo(os.path.join(DATA_PNG_PATH_PREFIX_MSIDETECT, str_trn_data, str_model))
            N_total = len(img_names)

            # labels
            Y_labels = get_label_vector(labels)

            # normalization 
            Xtmp = np.vstack([data_X[idx] for idx in range(N_total)])
            X_mean = np.mean(Xtmp, axis=0, keepdims=True)
            X_std = np.std(Xtmp, axis=0, keepdims=True)
            
            X_norm = [np.divide(data_X[idx] - X_mean, X_std) for idx in range(N_total)]   
            X_cs = data_feature_mat(sample_ids=img_names, data_mat=X_norm, Nis=Nis)

            # for class imbalance data: using the cost-sensitive loss                             
            n_pos = np.sum(Y_labels==1)
            n_neg = len(Y_labels) - n_pos

            setting.w_pos = n_neg/(n_pos + n_neg)                                           

            # train the DGP model
            str_trn_model_name = str_trn_data + '/MODEL_' + str(ith_run)
                
            model_ = dgp_rf_agg_ens(X_cs, Y_labels, setting, str_trndata=str_trn_model_name)                            
            model_.model_fit()
    else:
        setting.w_pos = 1.0 # dummy value
        
        seed_everything(11111)

    print("infernce model: vi ensemble")
     

    # evaluate test performance
    for str_tst_data in data_set_full:
        # print('tst data = ' + str_tst_data)

        if str_tst_data == str_trn_data:
            continue    
        else:                    
            str_save_path = os.path.join(RESULT_SAVE_PATH, 'predictions')

            if not os.path.exists(str_save_path):
                os.makedirs(str_save_path)        
            str_save_path = os.path.join(str_save_path, '_trn_' + str_trn_data + '_tst_' + str_tst_data)

            Yest_list = []            
            for ith_run in range(N_MSIDETECT_models):
                str_model = 'MODEL_' + str(ith_run)  

                str_save_each_model_path =  os.path.join(str_save_path, str_model)
                if not os.path.exists(str_save_each_model_path):
                    os.makedirs(str_save_each_model_path)   

                img_names_tst, img_tilenames_tst, X_tst, labels_tst, Ni_tst \
                        = get_MSI_data_with_tileinfo(os.path.join(DATA_PNG_PATH_PREFIX_MSIDETECT, str_tst_data, str_model))                                    
                 
                if ith_run == 0:
                    img_names_tst_ref = copy.deepcopy(img_names_tst)
                else:
                    #
                    idx_matched = [img_names_tst.index(elm) for elm in img_names_tst_ref]

                    X_tst = [X_tst[idx] for idx in idx_matched]
                    img_tilenames_tst = [img_tilenames_tst[idx] for idx in idx_matched]
                    labels_tst = [labels_tst[idx] for idx in idx_matched]

                    Ni_tst = Ni_tst[idx_matched]
                    
                Y_tst = get_label_vector(labels_tst)
                                
                N_tst = len(img_names_tst)
                idx_tst = np.array(range(N_tst))

                Xtmp = np.vstack([X_tst[idx] for idx in range(N_tst)])
                X_mean = np.mean(Xtmp, axis=0, keepdims=True)
                X_std = np.std(Xtmp, axis=0, keepdims=True)            
            
                X_tst_norm = [np.divide(X_tst[idx] - X_mean, X_std) for idx in range(N_tst)]

                X_tst_cs = data_feature_mat \
                (sample_ids=img_names_tst_ref, tile_names=img_tilenames_tst, data_mat=X_tst_norm, Nis=Ni_tst)
            
                # load the model  
                str_trn_model_name = str_trn_data + '/MODEL_' + str(ith_run)   

                # the test data is used only to create the class object  
                model_sepbest = dgp_rf_agg_ens(X_tst_cs, None, setting, str_trndata=str_trn_model_name)
        
                Ytst_probs_V1, _ = model_sepbest.predict\
                    (idx_tst, data_set_=X_tst_cs, save_pred_path=str_save_each_model_path)

                Yest_list.append(Ytst_probs_V1)

        # inference in ensemble learning
        Yest_all = []
        for mn_sub in range(N_tst):
            Yest_cur = []
            for ith_run in range(N_MSIDETECT_models):  # N_MSIDETECT_models
                Yest_cur.append(Yest_list[ith_run][mn_sub])

            Yest_all.append(np.hstack(Yest_cur))

        [Ytst_mean, unc_aleat, unc_epist] = cal_unc_quntities(Yest_all)
        BCS = 1 - 2*np.sqrt(unc_aleat + unc_epist)
        
        auc_val = roc_auc_score(Y_tst, Ytst_mean)
        print(f'test data={str_tst_data} (AUC={auc_val:.3f})')
        
    print("done")
else:
    print("End: no result")
