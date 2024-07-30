import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
import math

##    함수들
def columnFeature(column):
  uniqueLen = len(df[column].unique())
  uniqueLen = len(df[column].unique())
  valueCounts = df[column].value_counts(normalize=True)
  probabilities = valueCounts.values
  entropy = -np.sum(probabilities * np.log(probabilities))
  
  print(f'Max Entropy : {math.log(uniqueLen)}')
  print(f'Entropy: {entropy}')
  
  valueCounts.plot(kind='bar')
  plt.title('Unique Ratio')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.xticks(rotation=90)
  plt.show()
  
  plt.figure(figsize=(10, 6))
  sns.boxplot(x=df[column])
  plt.title('Box Plot')
  plt.xlabel('Amount')
  plt.show()

def preprocessNull(column, nullValue):
    colUnique = df[column].unique()
    uniqueDict = {value: index for index, value in enumerate(colUnique)}
    df[column] = df[column].fillna('N/A')
    uniqueDict['N/A'] = nullValue

    df[column] = df[column].map(uniqueDict)

def noiseAdd(column):
  noise = np.random.uniform(0, 0.5, len(df[column]))
  df[column] = df[column].where(df[column] == 0, df[column] + noise)
  df[column] = df[column].where(df[column] != 0, df[column] - noise)

##    unique 값 확인
df = pd.read_csv('D:/Kaggle/Spaceship_Titanic/Data/train.csv')
df['NullName'] = df['Name'].isnull().astype(int)
noiseAdd('NullName')
df['With'] = (df.groupby(df['PassengerId'].str[:4])['PassengerId'].transform('count'))-1

# qt = QuantileTransformer(output_distribution='uniform')
# df['With'] = qt.fit_transform(df[['With']])
power_transformer = PowerTransformer(method='yeo-johnson')
df['With'] = power_transformer.fit_transform(df[['With']])

# noise = np.random.uniform(0, 0.5, len(df['With']))
# df['With'] = df['With'].where(df['With'] != 0, df['With'] - noise)

# scaler = MinMaxScaler()
# df['With'] = scaler.fit_transform(df[['With']])
columns = df.columns

##    불리언 Null 값 처리하기(CryoSleep, VIP)
# preprocessNull('CryoSleep', -99)
# noiseAdd('CryoSleep')

df = df.drop(columns=['CryoSleep'])
preprocessNull('VIP', -1)
noiseAdd('VIP')
# df = df.drop(columns=['VIP'])

##    원 핫 인코딩(HomePlanet, Destination)
df['HomePlanet'] = df['HomePlanet'].fillna("N/A")
df['Destination'] = df['Destination'].fillna("N/A")
df = pd.get_dummies(df, columns=['HomePlanet'], prefix='Home')
df = pd.get_dummies(df, columns=['Destination'], prefix='Dest')

df[df.select_dtypes(include=[bool]).columns] = df.select_dtypes(include=[bool]).astype(int)

noiseCol = ['Home_Earth','Home_Europa','Home_Mars','Home_N/A','Dest_55 Cancri e','Dest_N/A','Dest_PSO J318.5-22','Dest_TRAPPIST-1e']
for col in noiseCol:
  noiseAdd(col)
df = df.drop(columns=['Home_N/A'])
df = df.drop(columns=['Dest_N/A'])

##    Age 열 처리
# columnFeature('Age')
avgAge = round(df['Age'].sum()/len(df['Age']), 5)
df['Age'] = df['Age'].fillna(avgAge)
scaler = MinMaxScaler()
df['Age'] = scaler.fit_transform(df[['Age']])
# qt = QuantileTransformer(output_distribution='uniform')
# df['Age'] = qt.fit_transform(df[['Age']])

##    RoomService, FoodCourt, ShoppingMall, Spa, VRDeck 열 처리
cashColumns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

for column in cashColumns:
  # moneyAvg = round((df[column].sum()) / len(df[column]), 5)
  # preprocessNull(column, moneyAvg)
  df[column] = df[column].fillna(-1)
  # scaler = RobustScaler()
  # df[column] = scaler.fit_transform(df[[column]])
  # qt = QuantileTransformer(output_distribution='uniform')
  # df[column] = qt.fit_transform(df[[column]])
  power_transformer = PowerTransformer(method='yeo-johnson')
  df[column] = power_transformer.fit_transform(df[[column]])

##    Cabin 열 처리
df['CabinNum'] = df['Cabin'].str.replace(r'[^0-9]', '', regex=True)
df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')
noiseAdd('CabinNum')
scaler = MinMaxScaler()
df['CabinNum'] = scaler.fit_transform(df[['CabinNum']])

df['Cabin'] = df['Cabin'].str.replace(r'[^A-Za-z]', '', regex=True)

df['Port_side'] = np.nan
df['Star_side'] = np.nan
df['Nan_side'] = 0

for index, row in df.iterrows():
    cabinVal = row['Cabin']
    
    if pd.isna(cabinVal):
        df.at[index, 'Nan_side'] = 1
    else:
        if 'P' in cabinVal:
            df.at[index, 'Port_side'] = cabinVal.replace('P', '')
        if 'S' in cabinVal:
            df.at[index, 'Star_side'] = cabinVal.replace('S', '')
            
df['Port_side'] = df['Port_side'].fillna(0)
df['Star_side'] = df['Star_side'].fillna(0)

df = pd.get_dummies(df, columns=['Port_side'], prefix='PS')
df = pd.get_dummies(df, columns=['Star_side'], prefix='SS')

df = df.drop(columns=['PS_0', 'SS_0'])
# df = df.drop(columns=['SS_T'])

noiseCol = ['PS_A','PS_B','PS_C','PS_D','PS_E','PS_F','PS_G','SS_A','SS_B','SS_C','SS_D','SS_E','SS_F','SS_G','PS_T','SS_T'] #,'PS_T','SS_T'
for col in noiseCol:
  noiseAdd(col)

df = df.drop(columns=['Cabin'])

##    기타
df['Name'] = df['Name'].fillna('Nobody')
df[df.select_dtypes(include=[bool]).columns] = df.select_dtypes(include=[bool]).astype(int)
# transportedColumn = df.pop('Transported')
# df['Transported'] = transportedColumn
# df = df.drop(columns=['RoomService'])
df = df.drop(columns=['Name'])
df = df.drop(columns=['PassengerId'])
df = df.drop(columns=['Nan_side'])
df.to_csv('D:/Kaggle/Spaceship_Titanic/Data/train_preprocess.csv', index=False)

"""
for col in noiseCol:
  # 'SS_T'가 양수인 행과 음수인 행의 수 계산
  positive_count = df[df[col] > 0].shape[0]
  negative_count = df[df[col] < 0].shape[0]

  # 전체 수
  total_count = positive_count + negative_count

  # 비율 계산
  positive_ratio = positive_count / total_count
  negative_ratio = negative_count / total_count

  # 결과 출력
  print(col)
  print(f"양수 비율: {positive_ratio:.2%} {positive_count}개")
  print(f"음수 비율: {negative_ratio:.2%} {negative_count}개")

##    상관관계
correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

O HomePlanet : ['Europa' 'Earth' 'Mars' nan]
O CryoSleep : [False True nan]
  Cabin : ['B/0/P' 'F/0/S' 'A/0/S' ... 'G/1499/S' 'G/1500/S' 'E/608/S']
O Destination : ['TRAPPIST-1e' 'PSO J318.5-22' '55 Cancri e' nan]
O  Age : [39. 24. 58. 33. 16. 44. 26. 28. 35. 14. 34. 45. 32. 48. 31. 27.  0.  1.
 49. 29. 10.  7. 21. 62. 15. 43. 47.  2. 20. 23. 30. 17. 55.  4. 19. 56.
 nan 25. 38. 36. 22. 18. 42. 37. 13.  8. 40.  3. 54.  9.  6. 64. 67. 61.
 50. 41. 57. 11. 52. 51. 46. 60. 63. 59.  5. 79. 68. 74. 12. 53. 65. 71.
 75. 70. 76. 78. 73. 66. 69. 72. 77.]
O VIP : [False True nan]
O  RoomService : [   0.  109.   43. ... 1569. 8586.  745.]
O  FoodCourt : [   0.    9. 3576. ... 3208. 6819. 4688.]
O  ShoppingMall : [   0.   25.  371. ... 1085.  510. 1872.]
O  Spa : [   0.  549. 6715. ... 2868. 1107. 1643.]
O  VRDeck : [   0.   44.   49. ... 1164.  971. 3235.]
Transported : [False  True]
"""