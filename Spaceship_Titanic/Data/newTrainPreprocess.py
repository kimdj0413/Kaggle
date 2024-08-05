import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer

def columnEntropy(column):
  max_entropy = np.log(len(df[column].unique()))
  value_counts = df[column].value_counts(normalize=True)
  entropy = -np.sum(value_counts * np.log(value_counts + 1e-10))
  plt.boxplot(df[column])
  plt.title(f'Box Plot of {column}')
  plt.ylabel(column)
  plt.show()
  print(f"최대 엔트로피: {max_entropy}")
  print(f"현재 엔트로피: {entropy}")

def nullRatio(column):
  df[column] = df[column].fillna(-1)
  filtered_df = df[df[column] == -1]
  transported_ratio = filtered_df['Transported'].mean()
  print(f"{column}가 Null인 경우의 'Transported' 비율: {transported_ratio:.2f}")

def scaleColumn(column, option):
  if option == "minmax":
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])
  elif option == "robust":
    scaler = RobustScaler()
    df[column] = scaler.fit_transform(df[[column]])
  elif option == "quantile":
    quantile = QuantileTransformer(output_distribution='uniform', random_state=42)
    row = df[[column]]
    quantileRow = quantile.fit_transform(row)
    df[column] = quantileRow
  elif option == "log":
    epsilon = 1e-10
    df[column] = np.log(df[column] + epsilon)
    
df = pd.read_csv('D:/Kaggle/Spaceship_Titanic/Data/train.csv')
intColumns = ['HomePlanet_Europa','HomePlanet_Mars','CryoSleep_True','Destination_PSO J318.5-22','Destination_TRAPPIST-1e','VIP_True','Cabin_S']
"""
['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age','VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','Name', 'Transported']
"""

df = df.drop(columns=['PassengerId','Name'])
df['Transported'] = df['Transported'].astype(int)

######################
##    HomePlanet    ##
######################
df = df.dropna(subset=['HomePlanet'])
df = pd.get_dummies(df, columns=['HomePlanet'], drop_first=True)

####################
##    CryoSleep   ##
####################
df = df.dropna(subset=['CryoSleep'])
df = pd.get_dummies(df, columns=['CryoSleep'], drop_first=True)

######################
##    Destination   ##
######################
df = df.dropna(subset=['Destination'])
df = pd.get_dummies(df, columns=['Destination'], drop_first=True)

##############
##    Age   ##
##############
df = df.dropna(subset=['Age'])
scaleColumn('Age','quantile')

##############
##    VIP   ##
##############
df = df.dropna(subset=['VIP'])
df = pd.get_dummies(df, columns=['VIP'], drop_first=True)

################
##    Cabin   ##
################
df = df.dropna(subset=['Cabin'])
df['Cabin'] = df['Cabin'].str[-1]
df = pd.get_dummies(df, columns=['Cabin'], drop_first=True)

################
##    Fare    ##
################
fareCol = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
for col in fareCol:
  df = df.dropna(subset=[col])
  scaleColumn(col,'quantile')

for col in intColumns:
  df[col] = df[col].astype(int)
print(df)

outlierIndex = [2860, 3135, 2569, 3574, 2722, 50, 101, 3113, 812, 2887, 6103, 2510, 1966, 503, 6032, 6001, 6378, 5159, 5117, 1894, 4584, 4619, 6176, 2845, 4894, 132, 6153, 6728, 3075, 4774, 811, 2057, 3159, 1315, 1893, 4850, 3145, 5907, 4280, 4071, 334, 5497, 4844, 239, 3049, 756, 1362, 31, 3718, 1039, 17, 5858, 3438, 2388, 4151, 6550, 2542, 5100, 752, 1650, 3822, 5891, 5890, 2499, 473, 2942, 4102, 
4526, 5978, 2284, 23, 5168, 4006, 833, 3833, 3872, 6541, 4689, 2742, 3946, 4781, 534, 6759, 3770, 1212, 2107, 5510, 2789, 79, 4216, 1595, 3428, 1840, 1242, 2498, 2893, 1183, 381, 5930, 2227, 5359, 5425, 2440, 457, 3765, 6094, 491, 4273, 1973, 4863, 1219, 1175, 3269, 2446, 1046, 3025, 535, 3996, 2272, 2083, 5654, 5623, 6361, 6570, 2653, 6479, 4641, 1483, 3196, 3132, 4180, 4995, 1406, 2348, 6245, 2705, 4716, 1961, 5379, 6289, 1732, 5192, 3235, 5886, 1407, 4369, 1726, 6556, 6132, 4954, 308, 1768, 2022, 1002, 1694, 5099, 1371, 6301, 3102, 4952, 1543, 3114, 2154, 2024, 1375, 6085, 1192, 2210, 2513, 4222, 6522, 696, 332, 2316, 2398, 2615, 6303, 2880, 5979, 6694, 2908, 5827, 5711, 3210, 2338, 6280, 3686, 1477, 2351, 3252, 6665, 5247, 1188, 6099, 486, 1511, 5420, 5332, 2225, 6607, 5455, 2018, 5326, 5057, 221, 3480, 6192, 5265, 4483, 6006, 2876, 1471, 1049, 543, 1345, 291, 1354, 3614, 2157, 747, 6257, 672, 4043, 5742, 3265, 2488, 168, 319, 3405, 506, 4353, 4877, 6498, 333, 6371, 6755, 2575, 3003, 259, 6732, 3395, 5636, 5312, 
5206, 3328, 4204, 2407, 805, 2117, 19, 3790, 5657, 3091, 2264, 2515, 4710, 5740, 3383, 5244, 2094, 527, 4240, 730, 2115, 5848, 3846, 2149, 2879, 122, 5673, 1557, 1199, 4387, 439, 3101, 4672, 6095, 59, 2087, 6202, 4965, 6375, 1385, 3626, 5830, 2787, 6635, 3558, 6112, 2232, 1357, 6650, 5619, 2847, 4401, 5001, 3984, 533, 3076, 3381, 33, 44, 2118, 3649, 26, 63, 1044, 227, 2166, 2825, 969, 4417, 1600, 3078, 5818, 252, 6627, 4653, 5544, 2759, 641, 6393, 5004, 1702, 6725, 6619, 3817, 783, 1018, 2183, 2522, 2405, 893, 
1883, 3945, 1498, 1703, 3885, 4084, 4984, 5785, 453, 1467, 1119, 834, 5627]

df = df.reset_index()
df = df.drop(index=outlierIndex)
df = df.drop(columns=['index'])

df.to_csv('D:/Kaggle/Spaceship_Titanic/Data/train_preprocess2.csv',index=False)