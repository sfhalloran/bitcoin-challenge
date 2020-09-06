import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import linear_model

data = pd.read_csv('/Users/sarahalloran/data-science-challenge/bitcoin.csv').drop(['time_period_start', 'time_period_end', 'time_open', 'time_close'], axis=1)
data.head()

def lookback(dataset, timesteps = 60):
    # this uses the shift method of pandas dataframes to shift all of the columns down one row
    # and then append to the original dataset
    data = dataset
    for i in range(1, timesteps):
        step_back = dataset.shift(i).reset_index()
        step_back.columns = ['index'] + [f'{column}_-{i}' for column in dataset.columns if column != 'index']
        data = data.reset_index().merge(step_back, on='index', ).drop('index', axis=1)
        
    return data.dropna()

dt = data.drop(columns=['volume_traded','trades_count'])
features = lookback(dt, timesteps=240)

target = features['price_high'].values
features = features.drop('price_high', axis=1).values

#decided not to use this 
scaler = MinMaxScaler()
scaled_features = pd.DataFrame(scaler.fit_transform(features))
scaled_target = scaler.fit_transform(target.reshape(-1,1))

br = linear_model.BayesianRidge()
br.fit(features, target)
br_pred = br.predict(features)
rmse_br = np.sqrt(np.mean(np.square((target - br_pred))))
