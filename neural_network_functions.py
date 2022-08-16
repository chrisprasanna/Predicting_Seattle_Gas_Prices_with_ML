#%%
import datetime as dt
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr as PCC

# %% 
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Defaults to 7
            verbose (bool): If True, prints a message for each validation loss improvement. Defaults to False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 0
            path (str): Path for the checkpoint to be saved to. Defaults to 'checkpoint.pt'
            trace_func (function): trace print function. Defaults to print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
#%%
def nn_train(model, model_name, epochs, data_train_loader, data_val_loader, opt, scheduler, target_scaler, device, plot=True):
    """
    This function performs the neural network training pipeline

    Args:
        model (PyTorch object): _description_
        model_name (str): model name. Options are 'lstm', 'darnn', or 'harhn'
        epochs (int): maximum number of training epochs
        data_train_loader (PyTorch DataLoader object): training data loader
        data_val_loader (PyTorch DataLoader object): validation data loader
        opt (PyTorch Optimizer object): neural network training optimizer 
        scheduler (PyTorch Scheduler object): learning rate scheduler 
        target_scaler (Normalizer object): Normalizer object to de-normalize target data for plotting
        device (str): training device (e.g., 'cpu' or 'cuda')
        plot (bool, optional): _description_. Defaults to True.
    """
    
    loss = nn.MSELoss()

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=50, verbose=True, path=f'{model_name}.pt') 

    for i in range(epochs):
        
        ### ========== TRAINING ========== ###
        mse_train = 0
        for batch_x, batch_y_h, batch_y in data_train_loader :
            
            # Extract data and initialize optimizer 
            batch_x = batch_x.to(device)  
            batch_y = batch_y.to(device)
            batch_y_h = batch_y_h.to(device)
            opt.zero_grad()
            
            # Forward prediction
            if model_name == 'lstm':
                h, c = model.init_hidden_internal(batch_x.shape[0])
                y_pred, h, c = model(batch_x, batch_y_h, h, c)
            elif model_name == 'darnn':
                y_pred, _, _ = model(batch_x, batch_y_h)
            elif model_name == 'harhn':
                y_pred = model(batch_x, batch_y_h)

            # Compute loss
            y_pred = y_pred.squeeze(1)        
            l = loss(y_pred, batch_y)
            
            # Backwards pass
            l.backward()
            
            # Collect training loss
            mse_train += l.item()*batch_x.shape[0]
            
            # Step optimizer
            opt.step()

        ### ========== Validation ========== ###
        with torch.no_grad():
            mse_val = 0
            preds = []
            true = []
            for batch_x, batch_y_h, batch_y in data_val_loader:

                # Extract data
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_y_h = batch_y_h.to(device)
                
                # Forward predictions
                if model_name == 'lstm':
                    h, c = model.init_hidden_internal(batch_x.shape[0])
                    output, h, c = model(batch_x, batch_y_h, h, c)
                elif model_name == 'darnn':
                    output, _, _ = model(batch_x, batch_y_h)
                elif model_name == 'harhn':
                    output = model(batch_x, batch_y_h)

                # Collect validation loss
                output = output.squeeze(1)
                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())
                mse_val += loss(output, batch_y).item()*batch_x.shape[0]
                
        preds = np.concatenate(preds)
        true = np.concatenate(true)

        # Learning rate scheduler
        scheduler.step(mse_val / data_val_loader.__len__())
        lr = opt.param_groups[0]['lr']

        # early_stopping needs the validation loss to check if it has decreased, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(mse_val / data_val_loader.__len__(), model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        print("Iter: ", i, "train: ", (mse_train / data_train_loader.__len__())**0.5, 
              "val: ", (mse_val / data_train_loader.__len__())**0.5, 
              "LR: ", lr
             )
        if (plot == True) and (i % 10 == 0):
            preds = target_scaler.inverse_transform(preds)
            true = target_scaler.inverse_transform(true)
            mse = mean_squared_error(true, preds)
            mae = mean_absolute_error(true, preds)
            print("mse: ", mse, "mae: ", mae)
            plt.figure(figsize=(12, 6))
            plt.plot(preds, label='predicted')
            plt.plot(true, marker=".", markersize=10, color='black', linestyle = 'None', label='actual')
            plt.legend(loc="upper left")
            plt.title(f'Validation Results for Epoch {i}')
            plt.show()
    return
#%%
def nn_eval(model, model_name, data_test_loader, target_scaler, device, cols):
    """
    This function performs the neural network evaluation protocol

    Args:
        model (PyTorch object): _description_
        model_name (str): model name. Options are 'lstm', 'darnn', or 'harhn'
        data_test_loader (PyTorch DataLoader object): test data loader
        target_scaler (Normalizer object): Normalizer object to de-normalize target data for plotting
        device (str): training device (e.g., 'cpu' or 'cuda')
        cols (list): list of feature names for feature importance plotting
        
    """
    
    with torch.no_grad():
        mse_val = 0
        loss = nn.MSELoss()
        preds = []
        true = []
        alphas = []
        betas = []
        for batch_x, batch_y_h, batch_y in data_test_loader:
            # Extract data
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_y_h = batch_y_h.to(device)
            
            # Forward prediction
            if model_name == 'lstm':
                h, c = model.init_hidden_internal(batch_x.shape[0])
                output, h, c = model(batch_x, batch_y_h, h, c)
            elif model_name == 'darnn':
                output, alpha, beta = model(batch_x, batch_y_h)
                alphas.append(alpha.detach().cpu().numpy())
                betas.append(beta.detach().cpu().numpy())
            elif model_name == 'harhn':
                output = model(batch_x, batch_y_h)
            
            # test loss
            output = output.squeeze(1)
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())
            mse_val += loss(torch.squeeze(output), batch_y).item()*batch_x.shape[0]
    preds = np.concatenate(preds)
    true = np.concatenate(true)
    
    # Collect attention weights
    if model_name == 'darnn':
        alphas = np.concatenate(alphas)
        betas = np.concatenate(betas)
    
    # De-normalize target data
    preds = target_scaler.inverse_transform(preds)
    true = target_scaler.inverse_transform(true)
    
    # Collect results
    mse = mean_squared_error(true, preds)
    mae = mean_absolute_error(true, preds)
    r2 = r2_score(true, preds)
    pcc, _ = PCC(true, preds)  
    err = true - preds
    
    # Time series plot
    plt.figure(figsize=(12, 6), facecolor=(1, 1, 1))
    plt.plot(preds, label='preds')
    plt.plot(true, marker=".", markersize=10, color='black', linestyle = 'None', label='actual')
    plt.legend(loc="upper left")
    plt.title(f'Test Results for {model_name.upper()} Model', fontsize=15)
    plt.xlabel('Samples', fontsize=12)
    plt.ylabel('Seattle Gas Price', fontsize=12)
    plt.show()
    
    # Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=(1, 1, 1))
    sns.regplot(ax=ax, x=true, y=preds)
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predictions', fontsize=12)
    ax.set_title(f'{model_name.upper()} Test Prediction Correlation \n R2 = {r2:.3f} \n PCC = {pcc:.3f}', fontsize=15)
    ax.set_aspect('equal', 'box')
    
    # Error Histogram
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=(1, 1, 1))
    sns.histplot(ax=ax, data=err, kde=True, bins=10)
    ax.set_xlabel('Prediction Errors (U.S. $)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{model_name.upper()} Testing Prediction Errors \n MSE = {mse:.3f} \n MAE = {mae:.3f}', fontsize=15)
        
    # Feature importance
    if model_name == 'darnn':
        alphas = alphas.mean(axis=0)
        betas = betas.mean(axis=0).squeeze()
        betas = betas[::-1]

        # Average attention weights of feature/timestep
        attn = np.zeros([len(alphas), len(betas)])
        for i in range(len(alphas)):
            for j in range(len(betas)):
                attn[i,j] = (alphas[i] + betas[j]) / 2
        
        # max and min attention for plotting and color thresholding                
        max_attn = np.amax(attn)
        min_attn = np.amin(attn)
        min_range = min_attn + (0.25 * (max_attn - min_attn))
        max_range = max_attn - (0.25 * (max_attn - min_attn))

        # Attention Weights Heatmap
        fig, ax = plt.subplots(figsize=(10, 10), facecolor=(1, 1, 1))
        im = ax.imshow(attn, cmap='rainbow')
        ax.set_xticks(np.arange(len(betas)))
        ax.set_yticks(np.arange(len(alphas)))
        ax.set_xticklabels(["t-"+str(i) for i in np.arange(len(betas), 0, -1)])
        ax.set_yticklabels(cols)
        for i in range(len(cols)):
            for j in range(len(betas)):
                val = round(attn[i, j], 3)
                if val < max_range and val > min_range:
                    ax.text(j, i, val,
                            ha="center", va="center", color="k")
                else:
                    text = ax.text(j, i, val,
                                   ha="center", va="center", color="w")
        ax.set_title(f"DA-RNN Attention Weights:\nImportance of Features and Timesteps")
        cbar = fig.colorbar(im, ax=ax, label='More Important \u2192')
        cbar.set_label(label='\n More Important \u2192',size='12')
        plt.show()
        
        # Feature Importance Bar Plot
        plt.figure(figsize=(8, 8), facecolor=(1, 1, 1))
        plt.title("DA-RNN Feature Importance\n(All Timesteps along Lookback Window)")
        plt.bar(range(len(cols)), alphas)
        plt.xticks(range(len(cols)), cols, rotation=90)
        plt.ylabel('Attention Weight')
    
    return mse, mae, r2, pcc, preds, true, alphas, betas
#%%
def nn_forecast(model, model_name, data, timesteps, n_timeseries, true, preds, x_scaler, y_his_scaler, target_scaler, device, dates, plot_range=10):
    """
    This function uses a trained and tested neural network to forecast the gas price for the next prediction period
    The previous predicted values and the newly forecasted value are plotted along with the actual data points

    Args:
        model (PyTorch object): _description_
        model_name (str): model name. Options are 'lstm', 'darnn', or 'harhn'
        data (numpy array): full dataset with features as columns and the target variable as the last column
        timesteps (int): length of the rolling lookback window
        n_timeseries (int): input size of the model aka the number of features
        true (numpy array): actual target variable data points
        preds (numpy array): predicted target variable data points
        x_scaler (Normalize object): Normalize object for the feature data
        y_his_scaler (Normalize object): Normalize object for the target history data
        target_scaler (Normalize object): Normalize object for the target data
        device (str): training device (e.g., 'cpu' or 'cuda')
        dates (pandas datetime object): date range to plot
        plot_range (int, optional): Data point indices to plot. Defaults to 10.
    """
    
    data = data.to_numpy()
    
    # last sequence of available data
    data_x_unseen = data[-timesteps:,:-1]
    y_hist_unseen = data[-timesteps:,-1]
    y_hist_unseen = np.expand_dims(y_hist_unseen, axis=1)
    
    # normalize data
    data_x_unseen = x_scaler.transform(data_x_unseen)
    y_hist_unseen = y_his_scaler.transform(y_hist_unseen)
    
    # convert numpy data to tensors
    x = torch.Tensor(data_x_unseen).float().to(device).unsqueeze(0)
    y_hist = torch.Tensor(y_hist_unseen).float().to(device).unsqueeze(0)

    # forward prediction
    model.eval()
    if model_name == 'lstm':
        h0, c0 = model.init_hidden_internal(x.shape[0])
        prediction, h, c = model(x, y_hist, h0, c0)
    elif model_name == 'darnn':
        prediction, _, _ = model(x, y_hist)
    elif model_name == 'harhn':
        prediction = model(x, y_hist)
        
    prediction = prediction.cpu().detach().numpy()

    # prepare plots
    
    # initialize
    to_plot_data_y_val = np.zeros(plot_range)
    to_plot_data_y_val_pred = np.zeros(plot_range)
    to_plot_data_y_test_pred = np.zeros(plot_range)

    # only plot within the specified range
    to_plot_data_y_val[:plot_range-1] = true[-plot_range+1:]
    to_plot_data_y_val_pred[:plot_range-1] = preds[-plot_range+1:]

    to_plot_data_y_test_pred[plot_range-1] = target_scaler.inverse_transform(prediction)

    # replace zeros with None
    to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
    to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

    # plot
    plot_date_test = dates[-plot_range+1:]
    next_week = plot_date_test[-1] + dt.timedelta(days=7)
    plot_date_test.append(next_week)

    fig = figure(figsize=(25, 5), dpi=80, facecolor=(1, 1, 1))
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(plot_date_test, to_plot_data_y_val, label="Actual prices", marker=".", markersize=10, color='black')
    plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10)
    plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price for next week", marker=".", markersize=20, color='red')
    plt.title(f"Predicted Seattle gas price of the next week ({model_name.upper()})")
    plt.ylabel('Dollars / Gallon')
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

    print(f'{model_name.upper()} Predicted Seattle Gas Price of the next week: ${to_plot_data_y_test_pred[plot_range-1]:.2f}')
    
    return fig, to_plot_data_y_test_pred[plot_range-1]