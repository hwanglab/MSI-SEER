"""
Training patch level prediction for baseline model
Some codes from: https://github.com/KatherLab/HIA
"""

from __future__ import print_function

import os
import random
import argparse
import time
from tqdm import tqdm
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models
from efficientnet_pytorch import EfficientNet

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn import metrics

# --- parser part ---#

# training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training in patch level')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=None, help='fold number (default: None)')
parser.add_argument('--model_name', type=str, choices=['resnet', 'efficientnet', 'shufflenet'], default='resnet', 
                    help='model architecture')
parser.add_argument('--freeze', type=float, default=0.5,
                    help='freeze ratio of layers')
parser.add_argument('--train_dir', type=str, default=None, 
                    help='train data csv file directory')
parser.add_argument('--test_dir', type=str, default=None, 
                    help='test data csv file directory')
parser.add_argument('--result_dir', type=str, default='results', 
                    help='results root directory')
parser.add_argument('--exp', type=str, default='train_Severance_3fold', 
                    help='experiment name for experiment folder directory')
parser.add_argument('--splits', type=str, default=None, 
                    help='data split csv file directory for k-fold cross validation')
parser.add_argument('--epochs', type=int, default=8,
                    help='number of epochs to train (default: 8)')
parser.add_argument('--patience', type=int, default=5,
                    help='number of patience for early stopping (default: 5)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay (default: 0.00001)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size (default: 128)')

args = parser.parse_args()


# --- utils part ---#
def seed_everything(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Seed number:", args.seed)
seed_everything(args.seed)


def get_value_from_key(d, key):
    
    values = [v for k, v in d.items() if k == key]
    if values:
        return values[0]
    return None    


def get_key_from_value(d, val):
    
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None   



# --- data part ---#
def png_loader(image_path):

    img = Image.open(image_path)
    img = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor()])(img)
    return img


def npy_loader(image_path):

    img = np.load(image_path, allow_pickle=True).astype(np.uint8)
    img = transforms.Compose([transforms.ToPILImage(),
                                transforms.ToTensor()])(img)
    return img

    
class TileImageDataset(Dataset):
    
    def __init__(self, dataframe):
        self.file_paths = dataframe['file_paths'].to_numpy()
        self.labels = dataframe['labels'].to_numpy()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        
        if 'png' in self.file_paths[index]:
            image = png_loader(self.file_paths[index])
        else:
            image = npy_loader(self.file_paths[index])
           

        return image, self.labels[index]
    
    
def get_label_vec(df, label_dict):

    for i in df.index:
        key = df.loc[i, 'labels']
        df.at[i, 'labels'] = label_dict[key]
        
    return df


# --- model part ---#
def create_model(model_name):

    model_ft = None

    if model_name == "resnet":
        model_ft = models.resnet18(pretrained = True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
                
    elif model_name == 'efficientnet':
        
        model_ft = EfficientNet.from_name('efficientnet-b7')
        model_ft.load_state_dict(torch.load('pretrained_weights/efficientnet-b7.pth'), strict=False)
        model_ft._fc = nn.Linear(in_features=2560, out_features=2, bias=True)
        
    elif model_name == 'shufflenet':
        model_ft = models.shufflenet_v2_x1_0(pretrained = True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        
    else:
        print("Invalid model name!")
        
    return model_ft


# --- training part ---#
class EarlyStopping:
    def __init__(self, patience = 5, stop_epoch = 10, verbose=False):

        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'\nEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss    
        
        
def train(model, train_dataloader, val_dataloader, optimizer, criterion, args, result_path, fold):
    
    start = time.time()
    print('Training start...!')        

    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []

    print(f"Max number of epochs: {args.epochs}")
    print("Setting up early stopping.")
    print("Min number of stop epoch: 10")
    print(f"Number of patience: {args.patience}")
    
    early_stopping = EarlyStopping(patience = args.patience, stop_epoch = 10, verbose = True)   
    
    for epoch in range(args.epochs):
        
        phase = 'train'
        print(f'Epoch: {epoch}/{args.epochs - 1}')
        model.train() 
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, y_hat = torch.max(outputs, 1)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(y_hat == labels.data)
                
        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(train_dataloader.dataset)        
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss) 
        
        print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print() 
        
        if val_dataloader:
            print('VALIDATION...\n')
            phase = 'val'    
            model.eval()        
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm(val_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)        
                with torch.set_grad_enabled(phase == 'train'):            
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, y_hat = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(y_hat == labels.data)        
            val_loss = running_loss / len(val_dataloader.dataset)
            val_acc = running_corrects.double() / len(val_dataloader.dataset)            
            val_acc_history.append(val_acc)
            val_loss_history.append(val_loss)
            print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, val_loss, val_acc)) 
            if fold == 'FULL':
                ckpt_name = os.path.join(result_path, "best_model.pt")
            else:
                ckpt_name = os.path.join(result_path, f"{fold}_best_model.pt")                         
            early_stopping(epoch, val_loss, model, ckpt_name = ckpt_name)
            if early_stopping.early_stop:
                print('-' * 30)
                print("The Validation Loss Didn't Decrease, Early Stopping!!")
                print('-' * 30)
                break
            print('-' * 30)
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return model, train_acc_history, train_loss_history, val_acc_history, val_loss_history


# --- validation part ---#
def validate(model, dataloaders):
    
    phase = 'test'
    model.eval()
    probsList = []    
    for inputs, labels in tqdm(dataloaders):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(phase == 'train'):            
            probs = nn.Softmax(dim=1)(model(inputs)) 
            probsList = probsList + probs.tolist()
    return probsList 


def cal_slide_level_prediction(tile_level_result_path, result_path, label_dict, fold = 0, clamMil = False):
    
    data = pd.read_csv(tile_level_result_path)
    patients = list(set(data['slide_ids']))
    keys = list(label_dict.keys())
    yProbDict = {}
    
    for index, key in enumerate(keys):
        
        patientsList = []
        yTrueList = []
        yTrueLabelList = []
        yProbList = []         
        keys_temp = keys.copy()
        keys_temp.remove(key)
        
        for patient in patients:
            patientsList.append(patient)
            data_temp = data.loc[data['slide_ids'] == patient]                        
            data_temp = data_temp.reset_index()            
            yTrueList.append(data_temp['labels'][0])
            yTrueLabelList.append(get_key_from_value(label_dict, data_temp['labels'][0]))                        
            if not clamMil:
                dl_pred = np.where(data_temp[keys_temp].lt(data_temp[key], axis=0).all(axis=1), True, False)
                dl_pred = list(dl_pred)
                true_count = dl_pred.count(True)            
                yProbList.append(true_count / len(dl_pred))
                
            else:
                yProbList.append(np.mean(data_temp[key])) 
                        
        yProbDict[key] = yProbList
    yProbDict = pd.DataFrame.from_dict(yProbDict)
    df = pd.DataFrame(list(zip(patientsList, yTrueList, yTrueLabelList)), columns =['slide_ids', 'labels', 'labels_name'])
    df = pd.concat([df, yProbDict], axis=1)  
    print("Saving slide level result file:")
    slide_level_result_path = os.path.join(result_path, f'{fold}_test_slide_level_prediction_results.csv')
    print(slide_level_result_path)    
    df.to_csv(slide_level_result_path, index = False)


# --- main part ---#
if __name__ == "__main__":
        
    result_path = f'{args.result_dir}/{args.exp}_{args.model_name}'

    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok = True)
        print("Result directory '%s' created successfully" %result_path)
    
    # labels are different from main DGP models (In DGPs, label_dict = {'MSS':0, 'MSIMUT':1})
    label_dict = {'MSIMUT':0, 'MSS':1}

    if args.k is not None:           
        # k-fold cross validation 
        print(f"reading 3-fold split...! Fold: {args.k}")
        kfold_df = pd.read_csv(args.splits, dtype={'slide_ids': 'str', 'tile_ids': 'str', 'labels': 'str', 'file_paths': 'str', 'kfold': 'float'})
        kfold_df = get_label_vec(kfold_df, label_dict=label_dict)

        k = args.k

        train_val_df = kfold_df[kfold_df['kfold'] != k]
        test_df = kfold_df[kfold_df['kfold'] == k]

    else:
        # inter evaluation
        print("reading train info...!")
        train_val_df = pd.read_csv(args.train_dir)
        
        if args.test_dir != None:
            test_df = pd.read_csv(args.test_dir)
            test_df = get_label_vec(test_df, label_dict=label_dict)

        train_val_df = get_label_vec(train_val_df, label_dict=label_dict)

        # if full train, k will set as 0
        k = 0

    train_val_df.reset_index(drop=True, inplace=True)

    train_df, val_df = train_test_split(train_val_df, random_state=args.seed, shuffle=True, test_size=0.1, stratify=train_val_df[['labels']])
    
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    print("saving splits information.")
    train_df.to_csv(os.path.join(result_path, f'{k}_train_split.csv'), index = False)
    val_df.to_csv(os.path.join(result_path, f'{k}_val_split.csv'), index = False)
    test_df.to_csv(os.path.join(result_path, f'{k}_test_split.csv'), index = False)
    

    training_data = TileImageDataset(dataframe=train_df)
    val_data = TileImageDataset(dataframe=val_df)

    print("creating tile datasetloader for training.")        
    # Preparing your data for training with DataLoaders
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    if (args.k != None) or (args.test_dir != None):
        print("creating tile datasetloader for testing.")        
        test_df.reset_index(drop=True, inplace=True)
        test_df.to_csv(os.path.join(result_path, f'{k}_test_split.csv'), index = False)
        test_data = TileImageDataset(dataframe=test_df) 
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    print(f"creating model: {args.model_name}")
    model = create_model(model_name=args.model_name)
    model = model.to(device)

    num_layers = 0
    for name, child in model.named_children():
         num_layers += 1            
    freezing_layers = int (args.freeze * num_layers)

    print(f"total number of layers: {num_layers}")
    print(f"total number of freezing layers: {freezing_layers}")

    print("freezing layers...")
    layers = 0
    for name, child in model.named_children():
        layers += 1
        if layers < freezing_layers:
            for child_name, params in child.named_parameters():
                params.requires_grad = False

    print("Initialize optimizer and loss function.")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr, weight_decay = args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    print("k:",k)

    model, train_acc_history, train_loss_history, val_acc_history, val_loss_history = train(model=model, 
                                                                                            train_dataloader=train_dataloader, 
                                                                                            val_dataloader=val_dataloader, 
                                                                                            optimizer=optimizer, 
                                                                                            criterion=criterion, 
                                                                                            args=args,
                                                                                            result_path=result_path,
                                                                                            fold = str(k))
    print("Saving final model.")
    torch.save(model.state_dict(), f'{result_path}/{k}_final_model.pt')
                     
                     
    history = pd.DataFrame(list(zip(train_loss_history, train_acc_history, val_acc_history, val_loss_history)), 
                      columns =['train_loss', 'train_acc', 'val_loss', 'val_acc'])
                     
    print("Saving training history file.")                     
    history.to_csv(os.path.join(result_path, f'{k}_train_history.csv'), index = False)
    
    
    if (args.k != None) or (args.test_dir != None):  
        
        print("Testing model..!")
        
        print("Loading best model.")                    
        model.load_state_dict(torch.load(os.path.join(result_path, f"{k}_best_model.pt")))                    
        probsList  = validate(model = model, dataloaders = test_dataloader)

        probs = {}
        for key in list(label_dict.keys()):
            probs[key] = []
            for item in probsList:
                probs[key].append(item[get_value_from_key(label_dict, key)])

        print("Saving tile level result file:")                                         
        probs = pd.DataFrame.from_dict(probs)
        tile_level_results = pd.concat([test_df, probs], axis = 1)                    
        tile_level_result_path = os.path.join(result_path, f'{k}_test_tile_level_prediction_results.csv')
        tile_level_results.to_csv(tile_level_result_path, index = False)
        print(tile_level_result_path)

        cal_slide_level_prediction(tile_level_result_path=tile_level_result_path, result_path=result_path, label_dict=label_dict, fold = k, clamMil = False)
    
    print("~ END ~")
