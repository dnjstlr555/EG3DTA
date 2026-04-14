import matplotlib.pyplot as plt
import pandas as pd
import os 
import numpy as np
import torch
import joblib
import torch.nn as nn
import time

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

class TrainLogger:
    def __init__(self, model, stdy, scheduler, prefix, patient=100, saveonperiod=True, saveperiod=200, warmup=40, printperiod=10):
        self.min_loss=9999
        self.min_pred=np.array([])
        self.min_label=np.array([])
        self.min_epoch=0
        self.history=[]
        self.stdy=stdy #loss 계산할때 필요.
        self.model=model #model 저장할때 필요 reference.
        self.scheduler=scheduler
        self.prefix=prefix
        self.patient=patient
        self.saveonperiod=saveonperiod
        self.saveperiod=saveperiod
        self.warmup=warmup
        self.printperiod = printperiod
    def step(self, ex, epoch, losses, valid_loss, labels, preds, lr, sblb=False, cvind=None):
        if sblb:
            ari = adjusted_rand_score(labels.reshape(-1), preds.reshape(-1).round())
            valid_loss = -ari
        mae, rmse, mape_value, r2 = losses_for_metrics(labels, preds, self.stdy)
        self.history.append([epoch, losses, valid_loss, mae, rmse, mape_value, r2, lr])
        if (self.warmup<epoch and self.min_loss>valid_loss) or (self.warmup>=epoch):
            self.min_loss = valid_loss
            self.min_pred = preds
            self.min_label = labels
            self.min_epoch = epoch
            print('Epoch: {}, Loss: {:.4f} Val: {:.4f} RMSE: {:.4f} MAE: {:.4f} MAP: {:.4f} R2:{:.4f} LR: {}'.format(epoch, losses, valid_loss, rmse, mae, mape_value, r2, self.scheduler._last_lr))
            if self.warmup<epoch or epoch==0:
                torch.save(self.model.state_dict(), '{}_ex{}_fold{}_best.pth'.format(self.prefix, ex, cvind))
        if epoch % self.printperiod == 0:
            if sblb:
                print('ARI: {:.4f}'.format(ari))
            print('Epoch: {}, Loss: {:.4f} Val: {:.4f} RMSE: {:.4f} MAE: {:.4f} MAP: {:.4f} R2:{:.4f} LR: {}'.format(epoch, losses, valid_loss, rmse, mae, mape_value, r2, self.scheduler._last_lr))
        if self.saveonperiod and (epoch % self.saveperiod == 0):
                torch.save(self.model.state_dict(), '{}_epoch_{}_ex{}.pth'.format(self.prefix, epoch, ex))
        if self.warmup<epoch and (epoch - self.min_epoch >= self.patient):
            return True
        return False
    def plot_history(self):
        plot_pred_label(self.min_pred, self.min_label)
        plot_loss([h[1] for h in self.history[5:]], [h[2] for h in self.history[5:]], [h[7] for h in self.history[5:]])

def losses_for_metrics(y_true, y_pred, stdy=None):
    #if pred or true have nan values, return np.inf for all metrics
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        return np.inf, np.inf, np.inf, -np.inf
    if stdy is None:
        y_true_c=y_true
        y_pred_c=y_pred
    else:
        y_true_c=stdy.inverse_transform(y_true)
        y_pred_c=stdy.inverse_transform(y_pred)

    # y_true and y_pred is tensor
    # Convert to numpy arrays for sklearn metrics
    mae = mean_absolute_error(y_true_c, y_pred_c)
    rmse = np.sqrt(mean_squared_error(y_true_c, y_pred_c))
    mape_value = mean_absolute_percentage_error(y_true_c, y_pred_c)
    r2 = r2_score(y_true_c, y_pred_c)
    return mae, rmse, mape_value, r2

def shape_normalize(data):
    '''total = []
    for i in range(len(data)):
        mydata = data[i]
        mydata = mydata - mydata[0:1, :, :]
        total.append(mydata)
    total = np.stack(total, axis=0)
    return total'''
    return data


def preprocess(data, sblb=False, plot_dist=True):
    #data의 i번째 fold에 대한 train, test 데이터를 리턴하는 함수
    train_x, train_y, test_x, test_y, valid_x, valid_y = shape_normalize(data[0]), data[1], shape_normalize(data[2]), data[3], shape_normalize(data[4]), data[5]
    train_x = train_x [:, :, :, :4]
    test_x = test_x [:, :, :, :4]
    valid_x = valid_x [:, :, :, :4]
    if plot_dist:
        plt.hist(train_y, bins=10, alpha=0.5, label='train')
        plt.hist(valid_y, bins=10, alpha=0.5, label='valid')
        plt.hist(test_y, bins=10, alpha=0.5, label='test')
        plt.title(f"Train/Test Labels Distribution")
        plt.xlabel("Labels")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
    
    if sblb:
        #use labelencoder for sublabels
        le = LabelEncoder()
        train_y = le.fit_transform(train_y)
        test_y = le.fit_transform(test_y)
        valid_y = le.fit_transform(valid_y)

    stdx=StandardScaler()
    stdy=StandardScaler()
    original_x_shape = train_x.shape
    
    train_x = train_x.reshape(-1, original_x_shape[2] * original_x_shape[3])
    test_x = test_x.reshape(-1, original_x_shape[2] * original_x_shape[3])
    valid_x = valid_x.reshape(-1, original_x_shape[2] * original_x_shape[3])
    
    train_x = stdx.fit_transform(train_x)
    test_x = stdx.transform(test_x)
    valid_x = stdx.transform(valid_x)
    
    train_x = train_x.reshape(-1, original_x_shape[1], original_x_shape[2], original_x_shape[3])
    test_x = test_x.reshape(-1, original_x_shape[1], original_x_shape[2], original_x_shape[3])
    valid_x = valid_x.reshape(-1, original_x_shape[1], original_x_shape[2], original_x_shape[3])
    
    train_y = stdy.fit_transform(np.expand_dims(train_y, axis=1))
    test_y = stdy.transform(np.expand_dims(test_y, axis=1))
    valid_y = stdy.transform(np.expand_dims(valid_y, axis=1))
    
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float()
    valid_x = torch.from_numpy(valid_x).float()
    valid_y = torch.from_numpy(valid_y).float()
    test_x = torch.from_numpy(test_x).float()
    test_y = torch.from_numpy(test_y).float()
    
    #train x -> [N, T,V, C] -> [N, C, T, V, 1]
    train_x = train_x.permute(0, 3, 1, 2).unsqueeze(4)
    test_x = test_x.permute(0, 3, 1, 2).unsqueeze(4)
    valid_x = valid_x.permute(0, 3, 1, 2).unsqueeze(4)
    
    return train_x, train_y, test_x, test_y, valid_x, valid_y, stdy

class MyDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

def get_loader(train_x, train_y, test_x, test_y, valid_x, valid_y, batch_size=32):
    #train_x: [N, C, T, V, 1], train_y: [N, 1]
    train_dataset = MyDataset(train_x, train_y)
    test_dataset = MyDataset(test_x, test_y)
    valid_dataset = MyDataset(valid_x, valid_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, valid_loader

def train_model(model, optimizer, criterion, train_loader, scheduler, device="cuda:1"):
    model.train()
    losses = 0
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses += loss.item()
    #scheduler.step()
    return losses / len(train_loader)

def test_model(model, test_loader, criterion, device="cuda:1"):
    model.eval()
    test_losses = 0
    preds = []
    targets = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data=data.to(device)
        target=target.to(device)
        with torch.no_grad():
            output = model(data)
            test_loss = criterion(output, target)
            test_losses += test_loss.item()
            output = output.cpu().numpy()
            preds.extend(output.reshape(-1))
            label = target.cpu().numpy()
            targets.extend(label.reshape(-1))
    preds = np.array(preds)
    targets = np.array(targets)
    return test_losses / len(test_loader), preds.reshape(-1, 1), targets.reshape(-1, 1)

def plot_pred_label(pred, label):
    df=pd.DataFrame({'pred': pred.reshape(-1), 'label': label.reshape(-1)})
    df=df.sort_values(by='label')
    plt.scatter(range(len(df)), df['pred'], label='pred', color='red')
    plt.scatter(range(len(df)), df['label'], label='label', color='blue')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Predictions vs Labels')
    plt.show()

def plot_loss(train_losses, test_losses, learning_rates):
    #plot train and test, and plot learning rates with dual axis
    fig, ax1 = plt.subplots()
    ax1.plot(train_losses, label='train loss', color='blue')
    ax1.plot(test_losses, label='test loss', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(learning_rates, label='learning rate', color='green')
    ax2.set_ylabel('Learning Rate')
    ax2.legend(loc='upper right')

    plt.title('Train and Test Loss with Learning Rate')
    plt.show()

def getdata(raw_data, exs=1, cv=False, sblb=False):
    #데이터셋의 운동 index를 받아서 train, test 데이터를 리턴하는 함수
    data=raw_data[exs]
    train_x=data['train_data']
    if sblb:
        train_y=data['train_sbs']
    else:
        train_y=data['train_labels']
    test_x=data['test_data']
    if sblb:
        test_y=data['test_sbs']
    else:
        test_y=data['test_labels']
    if cv:
        return train_x, train_y, test_x, test_y, test_x, test_y #placeholder for validation data
    val_x=data['val_data']
    val_y=data['val_labels']
    return train_x,train_y, test_x, test_y, val_x, val_y

def train_kimore(data_path, model_init, new_model, prefix="stgcn", lrs=0.001, batch_size=128, epochs=700, ex_only=[1,2,3,4,5], raw_data_func=lambda x:x, processed_data_func=lambda x:x, earlystop_patient=100, saveonperiod=True, savetestlabel=True, device_sp="cuda:1", inference_only=False):
    raw_data=joblib.load(data_path)
    prefix = os.path.join("./results", prefix)
    cv=False
    kfold_len = 1
    if 'kfold' in data_path:
        cv=True #[ex]['train_data'][cvind]
        kfold_len = len(raw_data[1]['train_data'])
        print(f'K-Fold CV with {kfold_len} folds')
    across_ex_history = {}
    across_ex_history_std = {}
    across_ex_history_inference = {} 
    torch.cuda.empty_cache()
    model = new_model()
    for ex in range(1, 11):        
        if ex not in ex_only:
            if ex_only == [1,2,3,4,5]:
                continue
            test_result = [-1, -1, -1, -1, -1, -1, -1]
            across_ex_history[ex] = test_result
            across_ex_history_std[ex] = test_result
            across_ex_history_inference[ex] = [-1, -1, -1, -1, -1, -1]
            continue
        totaldata = getdata(raw_data, ex, cv=cv)
        totaldata = raw_data_func(totaldata)
        if cv:
            test_result_arr = []
            inference_result_arr = []
            for cvind in range(kfold_len):
                print(f'Processing fold {cvind + 1}/{kfold_len} for exercise {ex}')
                crcvdata = [d[cvind] for d in totaldata]
                train_x, train_y, test_x, test_y, _, __, stdy = processed_data_func(preprocess(crcvdata))
                train_loader, test_loader, valid_loader = get_loader(train_x, train_y, test_x, test_y, test_x, test_y, batch_size=batch_size)
                optimizer, criterion, scheduler = model_init(model, lrs)
                logger = TrainLogger(model, stdy, scheduler, prefix, patient=earlystop_patient, saveonperiod=saveonperiod, warmup=0)
                traintime=0
                if not inference_only:
                    start_time=time.time()
                    for epoch in range(epochs):
                        losses = train_model(model, optimizer, criterion, train_loader, scheduler, device_sp)
                        valid_loss, preds, labels = test_model(model, valid_loader, criterion, device_sp)
                        #scheduler.step(valid_loss)
                        lr = optimizer.param_groups[0]['lr']
                        earlystopping = logger.step(ex, epoch, losses, valid_loss, labels, preds, lr, cvind=cvind)
                        if earlystopping:
                            break
                    end_time=time.time()
                    traintime=end_time-start_time
                model.load_state_dict(torch.load('{}_ex{}_fold{}_best.pth'.format(prefix, ex, cvind)))
                start_time=time.time()
                test_loss, preds, labels = test_model(model, test_loader, criterion, device_sp)
                end_time=time.time()
                testtime=end_time-start_time #초단위
                
                mae, rmse, mape_value, r2 = losses_for_metrics(labels, preds, stdy)
                print('Epoch {:.4f} Test Loss: {:.4f} RMSE: {:.4f} MAE: {:.4f} MAP: {:.4f} R2: {:.4f}'.format(logger.min_epoch, test_loss, rmse, mae, mape_value, r2))
                test_result = [cvind, logger.min_epoch, logger.min_loss, test_loss, rmse, mae, mape_value, r2]
                test_result_arr.append(test_result)
                inference_result = [cvind, traintime, len(train_loader.dataset), len(valid_loader.dataset), testtime, len(test_loader.dataset)]
                inference_result_arr.append(inference_result)
                #logger.plot_history()
                if savetestlabel:
                    labeldf = pd.DataFrame({"Label": labels.reshape(-1), "Pred": preds.reshape(-1)})
                    labeldf.to_csv('ex{}_cv{}.csv'.format(ex,cvind), index=False)
                del model
                torch.cuda.empty_cache()
                model = new_model()
            test_result_arr = pd.DataFrame(test_result_arr, columns=['Fold', 'Epoch', 'Val Loss', 'Test Loss', 'RMSE', 'MAE', 'MAPE', 'R2'])
            inference_result_arr = pd.DataFrame(inference_result_arr, columns=['Fold', 'Train Time (s)', 'Train Size', 'Valid Size', 'Test Time (s)', 'Test Size'])
            test_result_arr.to_csv(f'{prefix}_ex{ex}_cv_history.csv', index=False)
            inference_result_arr.to_csv(f'{prefix}_ex{ex}_cv_inference.csv', index=False)
            cvm=test_result_arr[['Epoch', 'Val Loss', 'Test Loss', 'RMSE', 'MAE', 'MAPE', 'R2']].mean()
            cvstd= test_result_arr[['Epoch', 'Val Loss', 'Test Loss', 'RMSE', 'MAE', 'MAPE', 'R2']].std()
            print(f'CV Mean: {cvm}')
            print(f'CV Std: {cvstd}')
            across_ex_history[ex] = cvm
            across_ex_history_std[ex] = cvstd
            across_ex_history_inference[ex] = inference_result_arr.mean()
        else:
            raise NotImplementedError("Non-CV mode is removed for simplicity.")
    df = pd.DataFrame(across_ex_history).T
    df.columns = ['Epoch', 'Val Loss', 'Test Loss', 'RMSE', 'MAE', 'MAPE', 'R2']
    df.to_csv(f'{prefix}_ex_history.csv', index=False)
    if cv:
        df_std = pd.DataFrame(across_ex_history_std).T
        df_std.columns = ['Epoch', 'Val Loss', 'Test Loss', 'RMSE', 'MAE', 'MAPE', 'R2']
        df_std.to_csv(f'{prefix}_ex_history_std.csv', index=False)
        df_inf = pd.DataFrame(across_ex_history_inference).T
        df_inf.columns = ['Fold', 'Train Time (s)', 'Train Size', 'Valid Size', 'Test Time (s)', 'Test Size']
        df_inf.to_csv(f'{prefix}_ex_history_inference.csv', index=False)
    print(df)
    print("All done!") #모든 작업 완료
