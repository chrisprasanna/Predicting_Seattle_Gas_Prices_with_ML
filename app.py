import streamlit as st
import pandas as pd
import numpy as np
import requests
import torch
from torch.utils.data import TensorDataset, DataLoader
from neural_network_classes import LSTM, DARNN, HARHN
from neural_network_functions import nn_eval, nn_forecast

# st.set_page_config(layout="wide")
st.write("# Seattle Gas Price Prediction App")

## load data
def download_data(url, name='', usecols=None, sheet_name=1, header=2): 
    """
    This function downloads and extracts relevant data from downloadable XLS files embedded on eia.gov

    Args:
        url (str): link address of the EIA XLS file
        name (str, optional): Name of the data variable. Defaults to ''.
        usecols (str, optional): XLS columns to extract (e.g., 'A:B'). Defaults to None.
        sheet_name (int, optional): Sheet number of the XLS file that contains the data. Defaults to 1.
        header (int, optional): How many rows of the XLS file are header files. Defaults to 2.
        plot (bool, optional): Option to plot the data variable. Defaults to False.

    Returns:
        dict: dictionary containing data, number of data points/elements, range of dates, and the data variable name
    """
    
    r = requests.get(url)
    open('temp.xls', 'wb').write(r.content)
    df = pd.read_excel('temp.xls', sheet_name=sheet_name, header=header, usecols=usecols) 
    df = df[~df.isnull().any(axis=1)] # remove rows with any missing data
       
    num_data_points = len(df)
    
    df2 = df.iloc[[0, -1]]    
    date_range = "from " + str(df2.iloc[0,0]) + " to " + str(df2.iloc[1,0])
    
    data_dict = {}
    data_dict['data'] = df.rename(columns={df.keys()[0]: 'date', 
                            df.keys()[1]: name})
    data_dict['num elements'] = num_data_points
    data_dict['date range'] = date_range
    data_dict['name'] = df.keys()[1]
    
    return data_dict

us_oil_stock = download_data('https://www.eia.gov/dnav/pet/hist_xls/MCRSCUS1m.xls',
                             name='oil stock exchange')
us_drilling_activity = download_data('https://www.eia.gov/dnav/pet/hist_xls/E_ERTRRG_XR0_NUS_Cm.xls',
                             name='drilling activity')
us_gas_production = download_data('https://www.eia.gov/dnav/ng/hist_xls/N9050US2m.xls',
                             name='gas production')
us_gas_consumption = download_data('https://www.eia.gov/dnav/ng/hist_xls/N9140US2m.xls',
                             name='gas consumption')
us_gas_storage = download_data('https://www.eia.gov/dnav/ng/xls/NG_STOR_CAP_DCU_NUS_M.xls',
                             name='gas storage',
                               usecols='A:B')
us_gas_import_volume = download_data('https://www.eia.gov/dnav/ng/xls/NG_MOVE_IMPC_S1_M.xls',
                             name='gas import volume',
                               usecols='A:B',
                               sheet_name=1)
us_gas_import_price = download_data('https://www.eia.gov/dnav/ng/xls/NG_MOVE_IMPC_S1_M.xls',
                             name='gas import price',
                               usecols='A:B',
                               sheet_name=2)
hh_natural_gas_price = download_data('https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDm.xls',
                             name='natural gas price')
us_crude_oil_price = download_data('https://www.eia.gov/dnav/pet/xls/PET_PRI_SPT_S1_M.xls',
                                   name='crude oil price',
                                   usecols='A:B',
                                   sheet_name=1)
us_gas_price = download_data('https://www.eia.gov/dnav/pet/xls/PET_PRI_SPT_S1_M.xls',
                                   name='conventional gas price',
                                   usecols='A:B',
                                   sheet_name=2)
us_rbob_price = download_data('https://www.eia.gov/dnav/pet/xls/PET_PRI_SPT_S1_M.xls',
                                   name='rbob gas price',
                                   usecols='A:B',
                                   sheet_name=3)
us_heating_oil_price = download_data('https://www.eia.gov/dnav/pet/xls/PET_PRI_SPT_S1_M.xls',
                                   name='heating oil price',
                                   usecols='A:B',
                                   sheet_name=4)
us_diesel_price = download_data('https://www.eia.gov/dnav/pet/xls/PET_PRI_SPT_S1_M.xls',
                                   name='diesel price',
                                   usecols='A:B',
                                   sheet_name=5)
us_kerosene_price = download_data('https://www.eia.gov/dnav/pet/xls/PET_PRI_SPT_S1_M.xls',
                                   name='kerosene price',
                                   usecols='A:B',
                                   sheet_name=6)
us_propane_price = download_data('https://www.eia.gov/dnav/pet/xls/PET_PRI_SPT_S1_M.xls',
                                   name='propane price',
                                   usecols='A:B',
                                   sheet_name=7)

# features
feature_list = [
    hh_natural_gas_price['data'],
    us_crude_oil_price['data'],
    us_gas_price['data'],
    us_rbob_price['data'],
    us_heating_oil_price['data'],
    us_diesel_price['data'],
    us_kerosene_price['data'],
    us_propane_price['data'],
    us_oil_stock['data'],
    us_drilling_activity['data'],
    us_gas_production['data'],
    us_gas_consumption['data'],
    us_gas_storage['data'],
    us_gas_import_volume['data'],
    us_gas_import_price['data']
]

# targets
seattle_gas_prices = download_data('https://www.eia.gov/dnav/pet/hist_xls/EMM_EPMRU_PTE_Y48SE_DPGw.xls', 
                                   name='gas price')
targets = seattle_gas_prices['data'].set_index('date')

# reindex and concatenate features
kw = dict(method="time") # method to interpolate feature data to target date indices 
for i in range(0, len(feature_list)):
    if i > 1:
        feature = feature_list[i].set_index('date')
        feature = feature.reindex(feature.index.union(targets.index)).interpolate(**kw).reindex(targets.index) # resample to target date indices
        features = features.join(feature)
    else:
        feature = pd.DataFrame(feature_list[i]).set_index('date') # initiate dataframe on first loop iteration
        features = feature.reindex(feature.index.union(targets.index)).interpolate(**kw).reindex(targets.index)  # resample to target date indices

# add week number (1-52) as a feature
features['week number'] = features.index.isocalendar().week
features['week number'] = features['week number'].astype(float)
    
# combine features and targets into one data frame
data = features.join(targets)

# get rid of rows with any missing data
data = data[~data.isnull().any(axis=1)]

# convert index datetimes to dates (exclude hour, minute, second)
data.index = data.index.date

## Lookback Window
target_name = 'gas price' # target variable name
feature_names = data.columns[0:-1].tolist() # exclude the last column aka the target variable

timesteps = 8 # rolling lookback window length

# Preallocate feature and target arrays
X_ = np.zeros((len(data), timesteps, data.shape[1]-1))
y_ = np.zeros((len(data), timesteps, 1))

# Include rolling lookback window sequences in feature array
for i, name in enumerate(list(data.columns[:-1])):
    for j in range(timesteps):
        X_[:, j, i] = data[name].shift(timesteps - j - 1).fillna(method="bfill")

# Include rolling lookback window sequences in target history array
for j in range(timesteps):
    y_[:, j, 0] = data[target_name].shift(timesteps - j - 1).fillna(method="bfill")
    
prediction_horizon = 1 # how far to train the neural nets to predict into the future 
target_ = data[target_name].shift(-prediction_horizon).fillna(method="ffill").values # shift target values 'prediction_horizon' times into the future 

## Normalize data
class Normalizer():
    def __init__(self):
        self.max = None
        self.min = None
        self.range = None

    def fit_transform(self, x):
        """
        This function computes the max, min and range of values from the training dataset
        These values will be used to normalize the training, validation, and testing datasets
        This function returns the normalized training dataset (values fall within [0:1])

        Args:
            x (numpy array): training data

        Returns:
            numpy array: _description_
        """
        self.max = x.max(axis=0)
        self.min = x.min(axis=0)
        self.range = self.max - self.min
        normalized_x = (x - self.min)/self.range
        return normalized_x
    
    def transform(self, x):
        """
        This function performs a normalization of the input dataset

        Args:
            x (numpy array): dataset that is being normalized based on the class instances 

        Returns:
            numpy array: normalized dataset
        """
        return (x - self.min)/self.range

    def inverse_transform(self, x):
        """
        This function performs a de-normalization of the input dataset

        Args:
            x (numpy array): dataset that is being de-normalized based on the class instances

        Returns:
            numpy array: de-normalized dataset
        """
        return (x*self.range) + self.min

up_to_train_idx = int(data.shape[0]*0.70)
train_length = up_to_train_idx
X = X_[timesteps:]
y = y_[timesteps:]
target = target_[timesteps:]
X_train = X[:train_length]
y_his_train = y[:train_length]
target_train = target[:train_length]

# Create normalizer class objects
x_scaler = Normalizer()
y_his_scaler = Normalizer()
target_scaler = Normalizer()

# Fit transforms
X_train = x_scaler.fit_transform(X_train)
y_his_train = y_his_scaler.fit_transform(y_his_train)
target_train = target_scaler.fit_transform(target_train)

## load model
model = DARNN(N=data.shape[1]-1, M=64, P=64,
              T=8, device='cpu')
model.load_state_dict(torch.load("./models/darnn.pt"))
model_name = 'darnn'

print('complete')

## APP

link = "[Source Code](https://github.com/chrisprasanna/Predicting_Seattle_Gas_Prices_with_ML)"
st.markdown(link, unsafe_allow_html=True)
# st.sidebar.write("[Source Code](https://github.com/chrisprasanna/Predicting_Seattle_Gas_Prices_with_ML)")

st.write("## 1. Data Overview")
if st.checkbox('Show Data'):
    data

# model_selection = st.sidebar.selectbox(
#         "Select a Neural Network Model",
#         [
#             "LSTM",
#             "DA-RNN",
#             "HARHN"
#         ]
#     )
st.write("## 2. Select a Model")
model_selection = st.selectbox(
        "",
        [
            "LSTM",
            "DA-RNN",
            "HARHN"
        ],
        index=1
    )
if model_selection == "LSTM":
    model = LSTM(num_outputs=1, input_size=data.shape[1], hidden_size=64, num_layers=2,
            seq_length=timesteps, device='cpu', dropout=0.2)
    model.load_state_dict(torch.load("./models/lstm.pt"))
    model_name = 'lstm'
elif model_selection == "DARNN":
    model = DARNN(N=data.shape[1]-1, M=64, P=64,
            T=8, device='cpu')
    model.load_state_dict(torch.load("./models/darnn.pt"))
    model_name = 'darnn'
elif model_selection == 'HARHN':
    model = HARHN(n_conv_layers=3, 
            T=timesteps, 
            in_feats=data.shape[1]-1, 
            target_feats=1, 
            n_units_enc=64, 
            n_units_dec=64, 
            device='cpu'
            )
    model.load_state_dict(torch.load("./models/harhn.pt"))
    model_name = 'harhn'

st.write("## 3. Select the Number of Data Points to Plot")
plot_length = st.slider('', min_value=5, max_value=int(len(data)-timesteps), value=10, step=1)

st.write("## 4. Launch Forecast")
if st.button('Make Prediction'):
    
    # Normalize
    X = x_scaler.transform(X)
    y = y_his_scaler.transform(y)
    target = target_scaler.transform(target)
    
    # Make Dataloader
    X_t = torch.Tensor(X[-plot_length:])
    y_t = torch.Tensor(y[-plot_length:])
    target_t = torch.Tensor(target[-plot_length:])
    data_loader = DataLoader(TensorDataset(X_t, y_t, target_t), shuffle=False, batch_size=32)
    
    # Get true and predicted time series
    mse, mae, r2, pcc, preds, true, _, _ = nn_eval(model=model, 
                                            model_name=model_name, 
                                            data_test_loader=data_loader, 
                                            target_scaler=target_scaler, 
                                            device='cpu', 
                                            cols=feature_names
                                            )
    
    # Forecast
    fig, prediction = nn_forecast(model = model,
            model_name = model_name, 
            data = data, 
            timesteps = timesteps, 
            n_timeseries = data.shape[1] - 1, 
            true = true, 
            preds = preds,
            x_scaler = x_scaler, 
            y_his_scaler = y_his_scaler, 
            target_scaler = target_scaler,
            device='cpu', 
            dates=data.index.tolist()[timesteps:],
            plot_range=plot_length
           )
    fig.set_size_inches(12, 4)
    fig.set_dpi(100)
    print("final pred", np.squeeze(prediction, -1))
    
    st.pyplot(fig, use_container_width=True)
    st.write(f"Next Week's Gas Price in Seattle: ${prediction:.2f}")
    
    col1, col2 = st.columns(2)
    col1.metric("Root-Mean-Square Error", f"{mse**0.5:.2f} $/gal")
    col2.metric("Mean Absolute Error", f"{mae:.2f} $/gal")
    col3, col4 = st.columns(2)
    col3.metric("Coefficient of Determination", f"{r2:.2f}")
    col4.metric("Pearson Correlation Coefficient", f"{pcc:.2f}")