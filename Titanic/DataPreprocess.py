import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import ADASYN

######################
##    함수 갈무리   ##
######################
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

def fillNa(column, option):
    if isinstance(option, int):
      df[column].fillna(option, inplace=True)
    else:
      if option == "drop":
        df[column] = df[column].dropna(inplace=True)
      
      elif option == "avg":
        meanVal = df[column].mean()
        df[column].fillna(meanVal, inplace=True)
        
      elif option == "most":
        mostVal = df[column].mode()[0]
        df[column].fillna(mostVal, inplace=True)
def noiseAdd(column):
  rng = np.random.default_rng(42)
  noise = rng.uniform(0, 0.5, len(df[column]))
  df[column] = df[column].where(df[column] == 0, df[column] + noise)
  df[column] = df[column].where(df[column] != 0, df[column] - noise)

####################################
##    데이터 로드 및 정보 확인    ##
###################################
df = pd.read_csv('D:/Kaggle/Titanic/Data/train.csv')
columns = df.columns
# print(df)
# print(columns)            12개
# print(len(df))            891행
# print(df.describe())      
# print(df.info())
"""
Index : ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
"""

################################
##    PassengerId 열 전처리   ##
################################
#   열 드롭
df = df.drop(columns=['PassengerId'])

############################
##    Pclass 열 전처리    ##
############################
#   유니크 값 확인
uniqueCnt= df['Pclass'].value_counts(normalize=True)
# print(uniqueCnt)
"""
원 핫 인코딩?
"""

##########################
##    Name 열 전처리    ##
#########################
#   열 드롭
df = df.drop(columns=['Name'])
"""
Mrs, Miss 처리?
"""

########################
##    Sex 열 전처리   ##
########################
#   원-핫 인코딩
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
df['Sex_male'] = df['Sex_male'].astype(int)

########################
##    Age 열 전처리   ##
########################
# 스케일링
fillNa('Age', -99)
scaleColumn('Age','quantile')

##########################
##    SibSp 열 전처리   ##
##########################
#   형제의 유무로
# df['YesSib'] = (df['SibSp'] == 0).astype(int)
# df = df.drop(columns=['SibSp'])
sibCnt = df['SibSp'].value_counts(normalize=True)
df['ZeroSib'] = (df['SibSp'] == 0).astype(int)
df['OneSib'] = (df['SibSp'] == 1).astype(int)
df = df.drop(columns=['SibSp'])
# print(sibCnt) 

##########################
##    Parch 열 전처리   ##
##########################
#   자녀 유무로
df['YesCh'] = (df['Parch'] == 0).astype(int)
df = df.drop(columns=['Parch'])

##############################
##    Embarked 열 전처리    ##
##############################
#   원 핫 인코딩
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
df['Embarked_Q'] = df['Embarked_Q'].astype(int)
df['Embarked_S'] = df['Embarked_S'].astype(int)

##########################
##    Fare 열 전처리    ##
##########################
scaleColumn('Fare','robust')

##########################
##    Cabin 열 전처리   ##
##########################
df['Cabin'] = df['Cabin'].str.replace(r'[^A-Za-z]', '', regex=True)
# df['manyCabin'] = 0
# df.loc[df['Cabin'].str.len() > 1, 'manyCabin'] += 1
df['Cabin'] = df['Cabin'].str[:1]
df['Cabin'].fillna("Z", inplace=True)
df = pd.get_dummies(df, columns=['Cabin'], drop_first=True)
cabinCol = ['Cabin_B','Cabin_C','Cabin_D','Cabin_E','Cabin_F','Cabin_G','Cabin_T','Cabin_Z']
for col in cabinCol:
  try:
    df[col] = df[col].astype(int)
  except KeyError:
    df[col] = 0
    print(f"{col} KeyError")
cabinZ = df.pop('Cabin_Z')
df['Cabin_Z'] = cabinZ


############################
##    Ticket 열 전처리    ##
############################
df = df.drop(columns=['Ticket'])
"""
# df['Ticket'] = df['Ticket'].apply(lambda ticket: ticket if ' ' in ticket else 'Number')
# # 띄어쓰기를 기준으로 나누고 첫 번째 원소만 남기기
# df['Ticket'] = df['Ticket'].str.split(' ').str[0]
# df['Ticket'] = df['Ticket'].replace({
#     # A 계열
#     'A/5': 'Aticket',
#     'A/5.': 'Aticket',
#     'A./5': 'Aticket',
#     'A.5': 'Aticket',
#     'A.5': 'Aticket',
#     'A./5.':'Aticket',
#     'A.5.':'Aticket',
#     'A4.': 'Aticket',
#     'A/4': 'Aticket',
#     'A/4.': 'Aticket',
#     'A/S':'Aticket',
    
#     # S.C. 계열
#     'S.C./A.4.': 'SCticket',
#     'S.C./PARIS': 'SCticket',
#     'SC/PARIS': 'SCticket',
#     'SC': 'SCticket',
#     'S.C.': 'SCticket',
#     'S.C./PARIS':'SCticket',
#     'SC/Paris':'SCticket',
#     'SC/AH':'SCticket',
    
#     # SOTON 계열
#     'SOTON/OQ': 'STticket',
#     'SOTON/O.Q.': 'STticket',
#     'SOTON/O': 'STticket',
#     'STON/O2.':'STticket',
#     'SOTON/O2':'STticket',
#     'STON/O':'STticket',
    
#     # W 계열
#     'W./C.': 'Wticket',
#     'W/C': 'Wticket',
#     'W.E.P.': 'Wticket',
#     'WE/P':'Wticket',
    
#     # PP 계열
#     'PP': 'Pticket',
#     'P/PP': 'Pticket',
#     'PC':'Pticket',
#     'SW/PP':'Pticket',
    
#     # F.C. 계열
#     'F.C.C.': 'Fticket',
#     'F.C.': 'Fticket',
#     'Fa':'Fticket',
    
#     # C 계열
#     'C.A.': 'Cticket',
#     'CA': 'Cticket',
#     'CA.': 'Cticket',
#     'C.A./SOTON':'Cticket',
#     'C':'Cticket',
      # SO 계열
#     'S.O.C.': 'Sticket',
#     'SO/C': 'Sticket',
#     'S.P.': 'Sticket',
#     'S.O.P.': 'Sticket',
#     'S.O./P.P.': 'Sticket',
#     'S.W./PP':'Sticket',
#     'SCO/W':'Sticket'
# })
# print(df['Ticket'].unique())
# df = pd.get_dummies(df, columns=['Ticket'], drop_first=True)
# ticketCol = ['Ticket_Cticket',
#        'Ticket_Fticket', 'Ticket_Number', 'Ticket_Pticket', 'Ticket_SCticket',
#        'Ticket_STticket', 'Ticket_Sticket', 'Ticket_Wticket']
# for col in ticketCol:
#   df[col] = df[col].astype(int)
"""

##############################
##    Survived 열 전처리    ##
##############################
"""
# ADASYN 언더샘플링
X = df.drop('Survived', axis=1)
y = df['Survived']

ada = ADASYN(sampling_strategy={1:1000, 0:1000}, random_state=42)
XSampling, ySampling = ada.fit_resample(X, y)
df = pd.DataFrame(XSampling, columns=X.columns)
df['Survived'] = ySampling
"""
print(df)
df.to_csv('./Titanic/Data/train_preprocess.csv', index=False)

"""
####################
##    난수 추가   ##
####################
# noiseCol = ['Pclass','Sex_male','ZeroSib','OneSib','YesCh','Embarked_Q','Embarked_S','Cabin_B','Cabin_C','Cabin_D','Cabin_E','Cabin_F','Cabin_G','Cabin_T','Cabin_Z']
# for col in noiseCol:
#   noiseAdd(col)
# print(df)


df.to_csv('./Titanic/Data/preprocess.csv')
####################
##    상관관계    ##
###################
df = df.drop(columns=['Ticket','Cabin'])
correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()
"""