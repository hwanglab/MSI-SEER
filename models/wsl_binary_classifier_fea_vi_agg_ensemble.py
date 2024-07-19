import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score
import copy

import os

from tqdm import tqdm
import pandas as pd

# --------------------------------------------------------------------------
EPS_ = 1e-16

mat_mul = torch.matmul
mat_add = torch.add

reduce_sum = torch.sum
reduce_mean = torch.mean

multiply = torch.multiply
divide = torch.divide

transpose = torch.transpose
squeeze = torch.squeeze

exp = torch.exp
log = lambda x: torch.log(x + EPS_)
logsumexp = torch.logsumexp

square = torch.square
sqrt = lambda x: torch.sqrt(x + EPS_)

sigmoid = torch.sigmoid
logistic_loss = lambda x: log(1.0 + exp(-x))

# https://pytorch.org/docs/stable/generated/torch.nn.ParameterList.html
torch.set_default_dtype(torch.float32)

device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
DROPOUT_PROB = 0.2
DROPOUT_1MP = 1.0 - DROPOUT_PROB

UNIF_R_BASE = 0.1

# deep Gausssian process with Random Feature expansion
class dgp_rf_(torch.nn.Module):
    def __init__(self, cnn_output_dim, setting, \
                 log_lw2=np.log(1.0), log_lb2=np.log(1.0), **kwargs):
        super(dgp_rf_, self).__init__(**kwargs)

        self.nMCsamples = setting.nMCsamples
        self.p_keep = DROPOUT_1MP

        self.n_GPlayers = setting.n_layers
        self.n_RFs = setting.n_RFs

        self.flag_norm = True

        self.ker_type = setting.ker_type

        self.input_dim = cnn_output_dim

        self.log_lw2_val = log_lw2
        self.log_lb2_val = log_lb2

        if self.ker_type == 'rbf':
            n_factor = 2
        else:
            n_factor = 1

        omega_input_dims = np.array([self.input_dim] * self.n_GPlayers)
        omega_input_dims[1:] = 2 * self.n_RFs

        w_dims = np.int_(np.append([self.n_RFs] * (self.n_GPlayers - 1), 2))

        self.Omegas = []
        self.W = []
        for idx in range(self.n_GPlayers):       
            O_layer = nn.Linear(in_features=omega_input_dims[idx], out_features=self.n_RFs, bias=False)
            W_layer = nn.Linear(in_features=n_factor * self.n_RFs, out_features=w_dims[idx], bias=True)

            torch.nn.init.xavier_uniform_(O_layer.weight)
            torch.nn.init.xavier_uniform_(W_layer.weight)
            W_layer.bias.data.fill_(0.01)

            self.Omegas.append(O_layer)
            self.W.append(W_layer)

        self.Omegas = nn.ModuleList(self.Omegas)
        self.W = nn.ModuleList(self.W)

        self.W_skip = nn.Linear(in_features=self.input_dim, out_features=self.n_RFs, bias=True)

        self.log_sigma2 = nn.Parameter(self.log_lw2_val * torch.ones(self.n_GPlayers), requires_grad=False)
        self.log_omega2 = nn.Parameter(self.log_lb2_val * torch.ones(self.n_GPlayers), requires_grad=False)

        self.dropout = nn.Dropout(p=DROPOUT_PROB, inplace=True)
        self.ReLU = nn.ReLU(inplace=True)

        return None

    # - Forward
    def forward(self, x):
        for mn_i in range(self.n_GPlayers):
            if mn_i == 0:
                x_in = x.repeat(self.nMCsamples, 1, 1)
                x_skip = self.W_skip(self.p_keep * self.dropout(x_in))
            else:
                x_in = F_out

            x_mdc = self.p_keep * self.dropout(x_in)
            Omega_x_ = self.Omegas[mn_i](x_mdc)

            if self.flag_norm and (mn_i > 2):
                mean_a = reduce_mean(Omega_x_, dim=-1, keepdims=True)
                std_a = sqrt(reduce_mean(square(Omega_x_ - mean_a), axis=-1, keepdims=True))
                Omega_x_ = divide(Omega_x_ - mean_a, std_a)

            if self.ker_type == 'rbf':
                factor_ = exp(self.log_sigma2[mn_i]) / np.sqrt(self.n_RFs)
                phi = factor_ * torch.concat((torch.cos(Omega_x_), torch.sin(Omega_x_)), axis=-1)
            else:
                factor_ = exp(self.log_sigma2[mn_i]) * (np.sqrt(2 / self.n_RFs))
                phi = factor_ * self.ReLU(Omega_x_)

            phi_mcd = self.p_keep * self.dropout(phi)
            F_out = self.W[mn_i](phi_mcd)

            if (mn_i < self.n_GPlayers - 1):
                F_out = torch.concat((F_out, x_skip), dim=-1)

        out_fs, out_logws = torch.split(F_out, split_size_or_sections=1, dim=-1)
        out_logws = transpose(squeeze(out_logws, dim=-1), 1, 0)

        return transpose(squeeze(out_fs, dim=-1), 1, 0), out_logws


    # - Rregularization
    def regularization(self):
        regu_loss = 0.0
        for mn_i in range(self.n_GPlayers):
            regu_loss += (self.p_keep * exp(self.log_omega2[mn_i]) * reduce_sum(square(self.Omegas[mn_i].weight)))
            regu_loss += (self.p_keep * reduce_sum(square(self.W[mn_i].weight)))

            regu_loss += reduce_sum(square(self.W[mn_i].bias))

        regu_loss += (self.p_keep * reduce_sum(square(self.W_skip.weight)))

        return regu_loss

    def set_model_hypParams(self, nMCsamples=None):
        if nMCsamples is not None:
            self.nMCsamples = nMCsamples

        return None
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------------
def chk_length(img_tile_names, prob_tiles):
    try:
        flg_chk = len(img_tile_names) == len(prob_tiles)
    except:
        flg_chk = False

    return flg_chk
                    
class wsl_classifier:
    def __init__(self, data_img, data_Y, setting, idx_trn=None, str_trndata=''):        
        # - data loader
        self.setting = setting                

        self.data_img = data_img
        self.Y = data_Y

        self.idx_trn = idx_trn                

        self.flag_ensemble = setting.exp_ensemble
        
        if self.flag_ensemble:            
            self.n_runs = setting.n_runs
        else:            
            self.n_runs = 1

            if self.idx_trn is None:
                print('single best model:: the training index should be given')
                
        # model save
        str_path = os.path.join(setting.save_directory, str_trndata)

        if not os.path.exists(str_path):
            os.makedirs(str_path)
        self.save_model_path = str_path

        return None

    def model_fit(self):   
        for ith_run in range(self.n_runs):
            print("--------------------------------------------------------------------------")
            if self.n_runs > 1:
                print(str(ith_run) + 'th run')
            
            if self.flag_ensemble:  
                if self.idx_trn is None:
                    trn_index_pos = np.random.choice(np.where(self.Y==1)[0], np.sum(self.Y==1), replace=True)
                    trn_index_neg = np.random.choice(np.where(self.Y==0)[0], np.sum(self.Y==0), replace=True)                                

                    trn_index = np.sort(np.concatenate((trn_index_pos, trn_index_neg)))

                    val_index = np.setdiff1d(np.array(range(len(self.Y))), trn_index)                
                else:                    
                    Y_ = np.reshape(self.Y[self.idx_trn], [-1])

                    trn_index_pos = np.random.choice(np.where(Y_==1)[0], np.sum(Y_==1), replace=True)
                    trn_index_neg = np.random.choice(np.where(Y_==0)[0], np.sum(Y_==0), replace=True)                                

                    trn_index = self.idx_trn[np.sort(np.concatenate((trn_index_pos, trn_index_neg)))]

                    val_index = np.setdiff1d(self.idx_trn, trn_index)                
            else:                
                trn_index = self.idx_trn
                val_index = None            
            
            # train each model
            str_model_save_path = os.path.join(self.save_model_path, "dgp_vi_" + str(ith_run))

            self.each_run = dgp_classifier(self.data_img, self.Y, self.setting, \
                trn_index=trn_index, val_index=val_index, str_filepath=str_model_save_path)
            
            self.each_run.model_fit()            

        return None

    def predict(self, tst_index, data_set_=None, sub_Ni=None, save_pred_path=''):
        if (save_pred_path != '') and (not os.path.exists(save_pred_path)):
            os.makedirs(save_pred_path)

        if sub_Ni is None:
            sub_Ni = self.setting.sub_Ni
                
        if data_set_ is None:
            data_set_ = self.data_img

        F_out = []
        W_out = []
        
        self.each_run = dgp_classifier(self.data_img, self.Y, self.setting, str_filepath=self.save_model_path)
        
        self.backup_nMCsamples = self.each_run.nMCsamples
        self.each_run.base_model.set_model_hypParams(nMCsamples=50)

        for ith_run in range(self.n_runs):
            str_model_load_path = os.path.join(self.save_model_path, "dgp_vi_" + str(ith_run))                
            self.each_run.load_model(str_model_load_path)

            F_sub, W_sub = self.each_run.predict(tst_index, data_set_, flag_trndata=False)

            F_out.append(F_sub)
            W_out.append(W_sub)

        #-
        n_runs = len(F_out)

        probs_Ver1 = []
        for dt_cnt, dt_idx in enumerate(tst_index):
            F_sub_list = [F_out[idx][dt_cnt] for idx in range(n_runs)]
            W_sub_list = [W_out[idx][dt_cnt] for idx in range(n_runs)]
                        
            # torch tensor
            F_sub = torch.tensor(np.hstack(F_sub_list))
            W_sub = torch.tensor(np.hstack(W_sub_list))

            if save_pred_path != '':
                # 
                str_img_id = data_set_.sample_ids[dt_idx]
                img_tile_names = data_set_.sample_tile_names[dt_idx]                
                                
                np.savez(os.path.join(save_pred_path, str_img_id + '_patch_maps.npz'), \
                                      img_tile_names, F_sub.cpu().detach().numpy(), W_sub.cpu().detach().numpy())

                # tile level predictions
                prob_tiles = reduce_mean(sigmoid(F_sub), dim=1, keepdim=True).cpu().detach().numpy()                
                weig_tiles = reduce_mean(tile_weight_norm(W_sub), dim=1, keepdim=True).cpu().detach().numpy()

                df_res = pd.DataFrame(np.column_stack((prob_tiles, weig_tiles)), columns=["P(tile=MSI-H)", "tile-pred-weight"])                            
                                              
                if chk_length(img_tile_names, prob_tiles):
                    df_res.index = img_tile_names
                    flag_index = True
                else:
                    # print(img_tile_names)
                    flag_index = False                     

                df_res.to_csv(os.path.join(save_pred_path, str_img_id + '_patch_probs.txt'),\
                     header=True, index=flag_index, mode='w', sep='\t', na_rep='NULL')

            # prediction probablities
            probs_pos = self.each_run.cal_BinProbs(F_sub, W_sub)
            probs_Ver1.append(probs_pos.cpu().detach().numpy())

        return probs_Ver1, 0.0        
# --------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------


mean_pbar = lambda x: reduce_mean(x, dim=1, keepdim=True)
mean_aleat = lambda x: reduce_mean(x - square(x), dim=1, keepdim=True)
mean_epist = lambda x, y: reduce_mean(square(x - y), dim=1, keepdim=True)

def cal_unc_quntities(p_hat):
    flag_direct_output = True

    if isinstance(p_hat, list):
        if not torch.is_tensor(p_hat[0]):
            flag_direct_output = False

            p_hat = [torch.Tensor(elm).to(device_) for elm in p_hat]

        p_bar = list(map(mean_pbar, p_hat))

        unc_aleat = torch.vstack(list(map(mean_aleat, p_hat)))
        unc_epist = torch.vstack(list(map(mean_epist, p_hat, p_bar)))

        p_bar = torch.vstack(p_bar)
    else:
        if not torch.is_tensor(p_hat):
            flag_direct_output = False
            
            p_hat = torch.Tensor(p_hat).to(device_)

        p_bar = reduce_mean(p_hat, dim=1, keepdim=True)

        unc_aleat = reduce_mean(p_hat - square(p_hat), dim=1, keepdim=True)
        unc_epist = reduce_mean(square(p_hat - p_bar), dim=1, keepdim=True)

    if flag_direct_output:
        return p_bar, unc_aleat, unc_epist
    else:
        return p_bar.cpu().detach().numpy(), unc_aleat.cpu().detach().numpy(), unc_epist.cpu().detach().numpy()

# weight normalization 
splif_fn_sum = lambda x: reduce_sum(x, dim=0, keepdim=True)
w_postive_fn = lambda x: UNIF_R_BASE + ((1.0 - UNIF_R_BASE) * sigmoid(x))

soft_max_0 = torch.nn.Softmax(dim=0)
def tile_weight_norm(w_est, Nis=None, weight_method='softmax'): #             
    w_est = w_postive_fn(w_est)

    if Nis is None:
        w_norm = divide(w_est, splif_fn_sum(w_est))
    else:
        w_est = torch.split(w_est, Nis, dim=0) 
        
        n_den = list(map(splif_fn_sum, w_est))  
        w_norm = torch.vstack(list(map(divide, w_est, n_den)))              

    return w_norm

# https://arxiv.org/pdf/1508.03422.pdf
#Y_true[Y_true==1] = w_pos*Y_true[Y_true==1]

loss_fun = lambda x: -log(torch.clamp(x, min=1e-7, max=1.0))
def approx_alpha_lik(f_est, w_tile, Nis, Y_true, N, w_pos=1.0, alpha=0.5, a_extrem=2.0):      
    ws_ = tile_weight_norm(w_tile, Nis=Nis)      
    F_w = torch.vstack(list(map(splif_fn_sum, multiply(ws_, f_est).split(Nis, dim=0))))
    
    probs_agg = sigmoid(a_extrem*multiply(Y_true, F_w))

    # cost-sensitive loss    
    # https://discuss.pytorch.org/t/use-class-weight-with-binary-cross-entropy-loss/125265/3
    # Yes, it would be calculated as nb_neg/nb_pos = 80/20 = 4.        
    # class_weights = torch.zeros_like(Y_true).to(device_)
    # class_weights[Y_true==1] = log(torch.from_numpy(np.array(w_pos, dtype=np.float32)).to(device_))

    class_weights = torch.zeros_like(Y_true).to(device_)

    class_weights[Y_true== 1] = torch.from_numpy(np.array(w_pos, dtype=np.float32)).to(device_)
    class_weights[Y_true==-1] = torch.from_numpy(np.array(1 - w_pos, dtype=np.float32)).to(device_)

    log_probs = logsumexp(-alpha*multiply(class_weights, loss_fun(probs_agg)), dim=1)                                            

    return -(N / (alpha*len(Y_true))) * reduce_sum(log_probs)


class dgp_classifier:
    def __init__(self, data_X, data_Y, setting, trn_index=None, val_index=None, str_filepath=None):                
        self.max_epoch = setting.max_epoch
        self.iter_print = setting.iter_print

        self.sub_Ni = setting.sub_Ni
        self.batch_size = setting.batch_size

        self.nMCsamples = setting.nMCsamples
        self.alpha = setting.alpha

        # - data loader
        self.data_X = data_X
        self.Y = data_Y
            
        self.trn_index = trn_index
        self.val_index = val_index

        self.model_save_full_path = str_filepath
        self.w_pos = setting.w_pos

        # - define the model        
        self.base_model = dgp_rf_(data_X.data_mat[0].shape[1], setting)
        self.base_model.to(device_)        

        self.sel_instances = setting.sel_instances # "sampling" 'none'

        # mean function 
        self.flag_mean_function = setting.flag_mean_function 
        if self.flag_mean_function:
            import re
            str_candid = re.findall("MODEL_\d", str_filepath)[0]
            model_number = np.int_(str_candid.split("_")[1])

            model_loading_path = os.path.join(setting.model_ref_path, f'MODEL_{model_number}_fc.pt')            
            model_ = torch.load(model_loading_path, map_location=torch.device(device_))    

            model_.eval()

            self.model_prior_mean = model_                        
            for param in self.model_prior_mean.parameters():
                param.requires_grad = False
        else:
            self.model_prior_mean = None

        if self.trn_index is not None:
            self.N = len(trn_index)
        else:
            return None

        # optimizer  
        self.minimum_epoch = np.minimum(self.max_epoch - 2, setting.minimum_epoch)
        self.optimizer = torch.optim.Adam(self.base_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07)

        return None

    def load_model(self, str_filefullpath):
        print('load the trained model: ' + str_filefullpath)

        self.base_model.load_state_dict(torch.load(str_filefullpath))
        return None

    def save_model(self, str_filefullpath):
        print('save the trained model: ' + str_filefullpath)

        torch.save(self.base_model.state_dict(), str_filefullpath)
        return None

    def model_fit(self):
        num_rep = np.int_(np.ceil(self.trn_index.size / self.batch_size))

        best_auc_val = 0.0
        for epoch in tqdm(range(self.max_epoch), desc='Training Epochs'):
            mv_trn_index = copy.deepcopy(self.trn_index)
            np.random.shuffle(mv_trn_index)

            mr_sumloss = 0.0
            mr_sumojb = 0.0
            for iter in range(num_rep):
                # load np data matrics
                index_vec = mv_trn_index \
                    [iter * self.batch_size:np.minimum(self.trn_index.size, (iter + 1) * self.batch_size)]

                if self.sel_instances == 'none':
                    X, _, Nis = self.gen_input_fromList(index_vec)
                else:
                    set_indices = self.mark_subImgs(index_vec, sub_Ni=self.sub_Ni)
                    X, _, Nis = self.gen_input_fromList(index_vec, set_indices)
            
                Y = (2 * (np.reshape(self.Y[index_vec], [-1, 1]) - 0.5)).astype(np.float32)

                X, Y = torch.Tensor(X).to(device_), torch.Tensor(Y).to(device_)                

                if self.flag_mean_function:
                    with torch.no_grad():
                        F_tmp = self.model_prior_mean(X)
                        F_prior = (F_tmp[:, 0] - F_tmp[:, 1]).reshape([-1, 1])
                
                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    F_est, W_est = self.base_model(X)
                    if self.flag_mean_function:                                                     
                        F_est = F_est + F_prior
                                        
                    loss = approx_alpha_lik(F_est, W_est, Nis, Y, self.N, self.w_pos)

                    mr_regul = self.base_model.regularization()
                    objective_fnc = loss + mr_regul

                    objective_fnc.backward()
                    self.optimizer.step()

                mr_sumloss += loss.item()
                mr_sumojb += objective_fnc.item()

            if self.val_index is not None:
                Y_est_val = self.predict(self.val_index, flag_trndata=True)
                auc_val = roc_auc_score(self.Y[self.val_index], Y_est_val)

                print('%3d:: trn obj = (%.3f, %.3f) & val_auc = (%.3f)' % \
                    (epoch, mr_sumloss / num_rep, mr_sumojb / num_rep, auc_val))

                if (epoch > self.minimum_epoch) and (auc_val > best_auc_val):
                    best_auc_val = auc_val

                    self.save_model(self.model_save_full_path)

        # the end of the epoch loop                    
        if self.val_index is None:
            self.save_model(self.model_save_full_path)

        return None

    def mark_subImgs(self, index_vec, sub_Ni):
        data_X = self.data_X

        # calculate the instnace weights
        Nis = np.hstack([data_X.Nis[idx] for idx in index_vec])

        _, W_est = self.model_eval_(data_X, index_vec, flag_trndata=True)
        with torch.no_grad():
            set_indices = []
            
            for mn_i, Ni in enumerate(Nis):            
                W_norm = reduce_mean\
                    (tile_weight_norm(W_est[mn_i]), dim=1).cpu().detach().numpy()
                
                idx_seq = np.array(range(Ni))
                if Ni > sub_Ni:
                    idx_selected = np.sort\
                        (np.random.choice(idx_seq, size=np.minimum(Ni, sub_Ni), p=W_norm/np.sum(W_norm), replace=False))                                            
                else:
                    idx_selected = idx_seq
                    
                set_indices.append(idx_selected)
            
        return set_indices 

    def gen_input_fromList(self, index_vec, set_indices=None):
        data_X = self.data_X

        Nis = []
        Nis_org = []
        for mn_i, idx in enumerate(index_vec, 0):
            if set_indices is None:
                Xsub = data_X.data_mat[idx]

                Ni = Xsub.shape[0]
            else:            
                idx_selected = set_indices[mn_i]
                Xsub = data_X.data_mat[idx][idx_selected]

                Ni = len(idx_selected)

            Nis.append(Ni)
            Nis_org.append(data_X.Nis[idx])

            if mn_i == 0:
                X = Xsub
            else:
                X = np.concatenate((X, Xsub), axis=0)

        X_idx = [cnt * np.ones((Ni, 1), dtype=np.int32) for cnt, Ni in enumerate(Nis, 0)]
        X_idx = np.reshape(np.vstack(X_idx), [-1])

        return X, X_idx, Nis

    # geometric mean
    # https://forum.effectivealtruism.org/posts/sMjcjnnpoAQCcedL2/when-pooling-forecasts-use-the-geometric-mean-of-odds#:~:text=Whereas%20the%20arithmetic%20mean%20adds,2%5D    
    def aggregate_probs(self, f_est, w_est, a_extrem=2.0):    
        w_norm = tile_weight_norm(w_est) 

        agg_probs = sigmoid(a_extrem * reduce_sum(multiply(w_norm, f_est), dim=0, keepdim=True)) 
        return agg_probs

    #
    def filter_out_rows_AnImage(self, outputs, mr_discardrate=0.025):
        Ni = outputs.shape[0]

        sorted_idx = torch.argsort(reduce_mean(outputs, dim=1), descending=False)

        mn_discard = np.int_(Ni * mr_discardrate)
        selected_idx = sorted_idx[range(mn_discard, Ni - mn_discard + 1)]

        return selected_idx
    
    def cal_BinProbs(self, F_out, W_out, n_rep=10, flag_trndata=False, selection='sampling'):
        if self.sel_instances == 'none':
            selection='none'                    

        with torch.no_grad():
            Ni = F_out.shape[0]
            
            if (flag_trndata) or (Ni <= self.sub_Ni):
                probs_out = self.aggregate_probs(F_out, W_out)

            else:                
                if selection == 'sampling':
                    probs_out = []        

                    W_norm = reduce_mean\
                        (tile_weight_norm(W_out), dim=1).cpu().detach().numpy()
                    
                    idx_seq = np.array(range(Ni))
                    for _ in range(n_rep):
                        selected_idx = np.sort(np.random.choice\
                                            (idx_seq, size=np.minimum(Ni, self.sub_Ni), p=W_norm/np.sum(W_norm), replace=False))                                        
                        
                        probs_out.append(self.aggregate_probs(F_out[selected_idx, :], W_out[selected_idx, :]))

                    probs_out = torch.hstack(probs_out)                            

                else:
                    selected_idx = self.filter_out_rows_AnImage(multiply(F_out, W_out))    
                    probs_out = self.aggregate_probs(F_out[selected_idx, :], W_out[selected_idx, :])

        return probs_out
    
    def predict(self, tst_index, data_set_=None, flag_trndata=True):
        if data_set_ is None:
            data_set_ = self.data_X

        # calcuate embeddings
        F_out, W_out = self.model_eval_\
            (data_set_, tst_index=tst_index, flag_trndata=flag_trndata)

        if flag_trndata == False:
            return F_out, W_out

        Probs_agg_out = []
        for cnt, _ in enumerate(tst_index):
            Probs_agg_est = self.cal_BinProbs(F_out[cnt], W_out[cnt], flag_trndata)
            Probs_agg_out.append(Probs_agg_est)

        return reduce_mean(torch.vstack(Probs_agg_out), dim=1, keepdim=True).cpu().detach().numpy()


    def model_eval_(self, data_set_, tst_index, flag_trndata=False):
        with torch.no_grad():
            F_out = []
            W_out = []
            for _, idx in enumerate(tst_index, 0):
                X = torch.Tensor(data_set_.data_mat[idx]).to(device_)
                F_est, W_est = self.base_model(X)

                if self.flag_mean_function:
                    F_tmp = self.model_prior_mean(X)
                    F_est = F_est + (F_tmp[:, 0] - F_tmp[:, 1]).reshape([-1, 1])
                
                if flag_trndata:
                    F_out.append(F_est)
                    W_out.append(W_est)
                else:
                    F_out.append(F_est.cpu().detach().numpy())
                    W_out.append(W_est.cpu().detach().numpy())

        return F_out, W_out
    