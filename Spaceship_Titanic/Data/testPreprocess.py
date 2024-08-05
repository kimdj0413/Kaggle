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
    
df = pd.read_csv('D:/Kaggle/Spaceship_Titanic/Data/test.csv')
intColumns = ['HomePlanet_Europa','HomePlanet_Mars','CryoSleep_True','Destination_PSO J318.5-22','Destination_TRAPPIST-1e','VIP_True','Cabin_S']

df = df.drop(columns=['PassengerId','Name'])
preDf = pd.read_csv('D:/Kaggle/Spaceship_Titanic/Data/test.csv')
quantile = QuantileTransformer(output_distribution='uniform', random_state=42)

######################
##    HomePlanet    ##
######################
df = pd.get_dummies(df, columns=['HomePlanet'], drop_first=True)

####################
##    CryoSleep   ##
####################
df = pd.get_dummies(df, columns=['CryoSleep'], drop_first=True)

######################
##    Destination   ##
######################
df = pd.get_dummies(df, columns=['Destination'], drop_first=True)

##############
##    Age   ##
##############
df = df.fillna(-99)
scaleColumn('Age','quantile')

##############
##    VIP   ##
##############
df = pd.get_dummies(df, columns=['VIP'], drop_first=True)

################
##    Cabin   ##
################
df['Cabin'] = df['Cabin'].str[-1]
df = pd.get_dummies(df, columns=['Cabin'], drop_first=True)

################
##    Fare    ##
################
fareCol = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
for col in fareCol:
  df = df.fillna(-99)
  scaleColumn(col,'quantile')

for col in intColumns:
  df[col] = df[col].astype(int)
print(df)
df = df.drop(columns='VIP_False')

df.to_csv('D:/Kaggle/Spaceship_Titanic/Data/test_preprocess2.csv',index=False)

print(df.isna().sum())