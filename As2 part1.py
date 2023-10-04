import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error as mse

def min_max_scaling(df):
    for column in df.columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())   
    return df

df = pd.read_csv('SteelPlateFaults-train.csv')
X_train = df[['Sum_of_Luminosity','Log_X_Index']]
x_train = min_max_scaling(X_train.copy())
y_train = df['LogOfAreas']
reg_train = LinearRegression().fit(x_train, y_train)#model

rmse_train = (mse(y_train, reg_train.predict(x_train))) ** 0.5#rmse calculation
print("RMSE for training data is", round(rmse_train, 5))



