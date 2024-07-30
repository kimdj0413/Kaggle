import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer
import math

df = pd.read_csv('D:/Kaggle/Spaceship_Titanic/Data/train.csv')

# #    각 열마다 이상치 제거
# print(df.describe())
# columns_to_check = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# for column in columns_to_check:
#     # lower_limit = df[column].quantile(0.01)
#     upper_limit = df[column].quantile(0.95)
#     df = df[(df[column] <= upper_limit)] #(df[column] >= lower_limit) & 
# print(df.describe())

##    열을 합쳐서 이상치 제거
# print(df.describe())
# print(len(df))
df['Amount'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
lower_limit = df['Amount'].quantile(0.03)
upper_limit = df['Amount'].quantile(0.93)
df = df[(df['Amount'] >= lower_limit)& (df['Amount'] <= upper_limit)] # & (df['Amount'] <= upper_limit)
df = df.drop(columns=['Amount'])
print(df.describe())
print(len(df))

df.to_csv('./Spaceship_Titanic/Data/train_sampling.csv', index=False)