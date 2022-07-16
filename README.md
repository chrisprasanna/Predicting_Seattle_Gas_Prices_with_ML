# :car::cloud_with_rain: Predicting Seattle Gas Prices using Machine Learning :chart_with_upwards_trend::chart_with_downwards_trend:

## Table of Contents <!-- omit in toc -->

- [Overview](#overview)
- [Installation](#installation)
- [Code Description](#code-description)
- [Model Comparisons](#model-comparisons)
- [To Do List](#to-do-list)
- [Credits](#credits)
- [License](#license)

## Overview

As of the summer of 2022, there has been a dramatic hike in the cost of retail gasoline. I live in Seattle, where the cost of living is alreday 53% higher than the national average, and I know a lot of individuals who have been hit especially hard due to higher energy costs. It would be very useful to find out if retail gas prices will increase, stay steady, or decrease a week in advance. Having access to this information may give some added flexibility to household budgets, particularly for those who have a large share of their income going toward essential costs such as transpotation, housing, and food.

In this project, I went through an end-to-end data science/machine learning workflow to create accurate models that predict retail gas prices in Seattle a week in advance. Two different types of forecasting techniques, time series and deep learning algorithms, were used for model development. For time series modeling, [Facebook Prophet](https://facebook.github.io/prophet/) and [NeuralProphet](https://github.com/ourownstory/neural_prophet) models were implemented. For deep learning modeling, [Long Short-Term Memory (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory), [Dual-stage Attention-based Recurrent Neural Network (DA-RNN)](https://arxiv.org/abs/1704.02971), and [Hierarchical attention-based Recurrent-Highway-Network (HRHN)](https://arxiv.org/abs/1806.00685) models were implemented. The models were trained using time series data from the [U.S. Energy Information Administration](https://www.eia.gov/) and the deep learning models were built using the [PyTorch](https://pytorch.org/) framework.

The machine learning (ML) workflow preesnted in this project goes through the following steps:

  1. [Source the time series data](#source-the-data)
  2. [Prepare the data for model training](#prepare-the-data)
  3. [Develop the models](#develop-the-models)
  4. [Train the models using a training subset of the prepared data](#train-the-models)
  5. [Evaluate the models on the testing subset of the prepared data](#evaluate-the-models)
  6. [Get future predictions (i.e., forecasts) from the models](#predict-the-future)

All these steps are included in the provided Jupyter Notebook. Neural network architecture classes and functions (train, evaluate, forecast) are provided in separate Python files within the directory.

All ready? Let's get to it!

## Installation
- Download this repository and move it to your desired working directory
- Download Anaconda if you haven't already
- Open the Anaconda Prompt
- Navigate to your working directory using the cd command
- Run the following command in the Anaconda prompt:
  
	```
  	conda env create --name NAME --file environment.yml
  	```

	where NAME needs to be changed to the name of the conda virtual environment for this project. This environment contains all the package installations and dependencies for this project.
  
- Run the following command in the Anaconda prompt:

  	```
  	conda activate NAME
  	```

	This activates the conda environment containing all the required packages and their versions.
  
- **For Jupyter Notebook Web Interface users:**
    - Open Anaconda Navigator
    - Under the "Applications On" dropdown menu, select the newly created conda environment
    - Install and open Jupyter Notebook. NOTE: once you complete this step and if you're on a Windows device, you can call the installed version of Jupyter Notebook within the conda environment directly from the start menu.  
    - Navigate to the Natural_Gas_Stock_Forecast.ipynb file within the repository
- **For VS Code users:**
    - [Install the Python extension](https://code.visualstudio.com/docs/python/python-tutorial) if you haven't already
    - Open the Command Palette (Ctrl+Shift+P)
    - Type *Python: Select Interpreter*
    - Select the newly-created Conda environment
    - Open Natural_Gas_Stock_Forecast.ipynb
    - Look at the top right
        - If you see the name of the Conda environment, you're done!
        - If you see a "Select Kernel" button, press it and select the Conda environment

## Code Description

This section goes over the essential tasks of the end-to-end ML pipeline. Code snippits are embedded inline.

### Source the Data

Before we can predict future gas prices, we'll need to acquire the historical time series data to train our models. For this, we will write Python code that downloads CSV files from [eia.gov](https://www.eia.gov/) and stores the data as variables. This way we won't have to worry about updating local CSV files on our machine with the most recent data. Instead, EIA does this for us. Thanks, government!

The following function uses the link address from EIA to download the CSV, parse through the data, and format the time series data as a Pandas DataFrame.
<!-- <details>
<summary>View codes</summary>

```python
def download_data(url, name='', usecols=None, sheet_name=1, header=2, plot=False): 
    global config
    
    r = requests.get(url)
    open('temp.xls', 'wb').write(r.content)
    df = pd.read_excel('temp.xls', sheet_name=sheet_name, header=header, usecols=usecols) 
    df = df[~df.isnull().any(axis=1)] # remove rows with any missing data
       
    num_data_points = len(df)
    
    df2 = df.iloc[[0, -1]]    
    date_range = "from " + str(df2.iloc[0,0]) + " to " + str(df2.iloc[1,0])
    print(date_range, str(num_data_points) + ' Data Points')
    
    data_dict = {}
    data_dict['data'] = df.rename(columns={df.keys()[0]: 'date', 
                            df.keys()[1]: name})
    data_dict['num elements'] = num_data_points
    data_dict['date range'] = date_range
    data_dict['name'] = df.keys()[1]
    
    if plot:
        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(data_dict['data']['date'], data_dict['data'][name], color=config["plots"]["color_actual"])
        plt.title(data_dict['name'] + ", " + data_dict['date range'] + ", " + str(data_dict['num elements']) + " Data Points")

        # Format the x axis
        locator = mdate.MonthLocator(interval=config["plots"]["xticks_interval"])
        fmt = mdate.DateFormatter('%Y-%m')
        X = plt.gca().xaxis
        X.set_major_locator(locator)
        # Specify formatter
        X.set_major_formatter(fmt)
        plt.xticks(rotation='vertical')
        plt.xlim([data_dict['data'].iloc[0,0], data_dict['data'].iloc[-1,0]])

        plt.grid(visible=None, which='major', axis='y', linestyle='--')
        plt.show()
         
    return data_dict
```
</details> -->

[Weekly Seattle retail gas price data (dollars/gallon)](https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=EMM_EPMRU_PTE_Y48SE_DPG&f=W) is used as the target variable, which is the variable whose values we will model and predict. A feature variable is a variable whose values will be used to help predict the future value of the target variable. Note that the time series models (Prophet and NeuralProphet) use the historical values of the target variable as their only feature variable. The reason for this is that these are univariate models. In other words, they are able to forecast long-term behavior using recursion but the trade-off is that they can only deal with one time series. Univariate models do not utilize feature variables since they do not have access to their future values and as a results. The prediction accuracy of these models may be limited since there exist a large number of other time series that could affect retail gas prices and could help provide insights to our predictions.

ML models are able to use feature variables as regressors, even if we do not have access to their future values. As long as we choose good external predictors as feature variables, we can expect the predictive performance of ML models to be superior to time series models. The downside is that we are only able to output predicted target varaible values based on the horizon that we train it on (e.g., we train the ML model to predict gas prices a week into the future). You can build a predictive model for each feature but keep in mind that using predicted values as features will propogate the error to the target variable. The features that were chosen for this project are as follows:

- U.S. Crude Oil Stock Change (Thousands Barrels)
- U.S. Natural Gas Rotary Rigs in Operation (Number of Elements)
- U.S. Natural Gas Production (Million Cubic Feet)
- U.S. Natural Gas Consumption (Million Cubic Feet)
- U.S. Underground Natural Gas Storage Capacity (Capacity in Million Cubic Feet)
- U.S. Natural Gas Import Volumes (Million Cubic Feet)
- U.S. Natural Gas Import Prices (Dollars/Thousand Cubic Feet)
- Henry Hub Natural Gas Spot Price (Dollars/Million Btu)
- Crude Oil Spot Price (Dollars/Barrel)
- Conventional Gasoline Price (Dollars/Gallon)
- RBOB Regular Gasoline Price (Dollars/Gallon)
- Heating Oil Price (Dollars/Gallon)
- Ultra-Low-Sulfur No. 2 Diesel Fuel Price (Dollars/Gallon)
- Kerosene-Type Jet Fuel Price (Dollars/Gallon)
- Propane Price (Dollars/Gallon)

Note that other relevant facors can be included as feature variables! This project can be expanded on by including data exploration pipelines. For instance, studying the correlation between variables may be insightful. Furthermore, studying the behavior of variables in response to significant events (e.g., natural disasters, war, pandemics, recessions, etc.) may allow us choose features that can best help our models make accurate predictions during periods of sudden change.

### Prepare the Data

Now that our program has downloaded the data, the next step is to prepare it for model training. The first step is to add another dimension to the time series data. Within this dimension we will add sequences containg a sequence of the past 4 weeks of the time series values. This method is called the rolling window method (sometimes called the lag method in statistics) and the window is different for every data point. Here is a great gif that demonstrates this method:

![Alt Text](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/11/3hotmk.gif)

Note that we will implement a rolling window on the target variable as well as all the feature variables. In practice, choosing a window length appropriate to the temporal dependencies of the problem greatly simplifies training and performance of the network. Too short of a window length increases the chance that the model parameter estimates do not produce high-fidelity predictions. On the other hand, too long of a window length increases the chance that you are trying to stretch your model to cover more cases than it can accurately represent. For this application, including sequences up to 4 weeks into the past seems to work well.

<!-- <details>
<summary>View codes</summary>

```python
timesteps = 4 # 10, lookback window

# Preallocate feature and target arrays
X_ = np.zeros((len(data), timesteps, data.shape[1]-1))
y_ = np.zeros((len(data), timesteps, 1))

# Feature Variables
for i, name in enumerate(list(data.columns[:-1])):
    for j in range(timesteps):
        X_[:, j, i] = data[name].shift(timesteps - j - 1).fillna(method="bfill")

# Historical Target Variable Values
for j in range(timesteps):
    y_[:, j, 0] = data[target_name].shift(timesteps - j - 1).fillna(method="bfill")

# Target Variables
prediction_horizon = 1
target_ = data[target_name].shift(-prediction_horizon).fillna(method="ffill").values
```
</details> -->

After transforming the dataset, the shape of our feature variable array `X_` is `(835, 4, 15)`, where the first dimension represents the 835 datapoints (i.e., timesteps), the second dimension represents the 4 historical values at each datapoint, and the third dimension represents the particular time series for each of the 15 feature variables. Note that the historical target variable values are kept separate in the `y_` array. The target variable values at each of the 835 datapoints are stored in the `target_` variable.

The next step is to split the dataset into three parts: training, validation, and testing. The training dataset is used to train the model and adjust its parameters. In other words, the model sees and learns from this data. The validation dataset is used to periodically evaluate the model during training. The models "see" this data, but they never learn from it. This dataset provides an unbiased evaluation of the model fit on the training dataset while tuning model hyperparameters. It is also useful for implementing callbacks such as early stopping and learning rate schedulers, which can help prevent overfitting. The test dataset is used to provide an unbiased evaluation of the final model fit on the training dataset. It is only used once the models are completely trained and let's us compare the performance of competing models. I split the data into a 70-15-15 split where 70% of the data is used foor training, the next 15% used for validation, and the remaining 15% for testing.

<!-- <details>
<summary>View codes</summary>

```python
# Dataset indices
up_to_train_idx = int(data.shape[0]*0.70)
up_to_val_idx = int(data.shape[0]*0.85)

# Number of data points in each dataset
train_length = up_to_train_idx
val_length = up_to_val_idx - up_to_train_idx
test_length = data.shape[0] - train_length - val_length

print(train_length, val_length, test_length)

X = X_[timesteps:]
y = y_[timesteps:]
target = target_[timesteps:]

X_train = X[:train_length]
y_his_train = y[:train_length]
X_val = X[train_length:train_length+val_length]
y_his_val = y[train_length:train_length+val_length]
X_test = X[-val_length:]
y_his_test = y[-val_length:]
target_train = target[:train_length]
target_val = target[train_length:train_length+val_length]
target_test = target[-val_length:]
```
</details> -->

![split](.images/../images/train-val-split.png)

The next data preparation step is to normalize the time series data. Since we are dealing with a number of feature variables with different units and a wide range of magnitudes, it is important that no single variable steers the model behavior in a particular direction just biggest it contains bigger numbers. The goal of normalization is to change the values of all the time series to a common scale, without distorting the differences in the shape of each time series. To normalize the machine learning model, values are shifted and rescaled so their range can vary between 0 and 1.

<!-- <details>
<summary>View codes</summary>

```python
class Normalizer():
    def __init__(self):
        self.max = None
        self.min = None
        self.range = None

    def fit_transform(self, x):
        self.max = x.max(axis=0)
        self.min = x.min(axis=0)
        self.range = self.max - self.min
        normalized_x = (x - self.min)/self.range
        return normalized_x
    
    def transform(self, x):
        return (x - self.min)/self.range

    def inverse_transform(self, x):
        return (x*self.range) + self.min

x_scaler = Normalizer()
y_his_scaler = Normalizer()
target_scaler = Normalizer()

X_train = x_scaler.fit_transform(X_train)
X_val = x_scaler.transform(X_val)
X_test = x_scaler.transform(X_test)

y_his_train = y_his_scaler.fit_transform(y_his_train)
y_his_val = y_his_scaler.transform(y_his_val)
y_his_test = y_his_scaler.transform(y_his_test)

target_train = target_scaler.fit_transform(target_train)
target_val = target_scaler.transform(target_val)
target_test = target_scaler.transform(target_test)
```
</details> -->

Notice that we use the maximum and minimum parameters from the trianing dataset to normalize both the validation and test dataset. This is because the validation and testing datapoints represent real-world data that we are using to evaluate our models. By using the training dataset normalization parameters on all three datasets, we can see how well each model generalizes to new, unseen datapoints. 

The final step is to convert the arrays into PyTorch tensor datasets and then translate the datasets into PyTorch DataLoader classes, which can iterate over a dataset during training. In addition, they are able to collect individually collected data samples and automatically convert them into batches. 

<!-- <details>
<summary>View codes</summary>

```python
X_train_t = torch.Tensor(X_train)
X_val_t = torch.Tensor(X_val)
X_test_t = torch.Tensor(X_test)
y_his_train_t = torch.Tensor(y_his_train)
y_his_val_t = torch.Tensor(y_his_val)
y_his_test_t = torch.Tensor(y_his_test)
target_train_t = torch.Tensor(target_train)
target_val_t = torch.Tensor(target_val)
target_test_t = torch.Tensor(target_test)

data_train_loader = DataLoader(TensorDataset(X_train_t, y_his_train_t, target_train_t), shuffle=True, batch_size=batch_size)
data_val_loader = DataLoader(TensorDataset(X_val_t, y_his_val_t, target_val_t), shuffle=False, batch_size=batch_size)
data_test_loader = DataLoader(TensorDataset(X_test_t, y_his_test_t, target_test_t), shuffle=False, batch_size=batch_size)
```
</details> -->

### Develop the Models

A total of five models were developed for this project. Let's start with the time series models. The first one is Prophet, a model that Facebook/Meta developed that decomposes time series into various time-varying components (e.g., daily seasonality, weekly seasonality, yearly seasonality, trend, and holidays/events). In essence, this is an additive model that sums different linear regressors. But what if we want to use nonlinear regressors?

NeuralProphet was developed to extend the Prophet model with an autoregressive network (AR-Net). This AR-Net term uses a feedforward neural network to learn an autoregressive model that uses local context information from the time series to add lagged covariates. NeuralProphet also includes all the components from the original Prophet model as regressors. NeuralProphet is more expressive than its predecessor and it also scales much better to larger datasets (i.e., its training time complexity is nearly constant as the number of model inputs increases compared to classic autoregresion that scales exponentially). The downside of NeuralProphet is that it includes a much larger number of learnable parameters and as a result, you need a lot more taining data.

Since Prophet and NeuralProphet are open source projects, I would suggest checking out their own documentation if you need any more details. Both models also automatically tune hyperparameters, which makes implementation very easy. Of course, if you want to manually tune hyperparameters yourself, you have the ability to do so. One thing I will note that is these time series models work best with time series that have strong seasonal effects. Retail gas prices due exhibit some seasonal behavior (e.g., gas prices rise in the summer and lower in the winter). However, are there other external factors or irregular events that can significantly affect gas prices?

Let's move to the deep neural networks, where we will have to do a bit of development ourselves using PyTorch. 

### Train the Models

### Evaluate the Models

### Predict the Future

## Model Comparisons

## To Do List

- Deploy models to a web app using Flask
- Create full time series prediction visualization
- Combine histographs of deep neural network models

## Credits

## License
[MIT](https://github.com/chrisprasanna/Predicting_Seattle_Gas_Prices_with_ML/blob/main/LICENSE.md)
