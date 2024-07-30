import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb

# 데이터 로드 및 전처리 (여기서는 위의 전처리 코드를 사용합니다)
df = pd.read_csv('D:/Kaggle/Spaceship_Titanic/Data/train_preprocess.csv')
# ... (여기에 전처리 과정을 추가하세요)

# 예시로 전처리된 데이터에서 'Transported' 열을 제외한 X와 y를 준비합니다.
X = df.drop(columns=['Transported'])
y = df['Transported']

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 기본 모델 정의
base_models = [
    ('rf', RandomForestClassifier(
        random_state=42,
        n_estimators=150,
        max_features='sqrt',
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
    )),
    ('xgb', xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        n_estimators=300,
        learning_rate=0.01,
        max_depth=7,
        subsample=0.6,
        colsample_bytree=0.6,
    )),
    ('lr', LogisticRegression(
        solver='liblinear'
    )),
    ('svm', SVC(
        random_state=42,
        C=100,
        gamma='auto',
        kernel='rbf'
    ))
]

# 메타 모델 정의
meta_model = LogisticRegression(solver='liblinear')

# 스태킹 모델 정의
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# 스태킹 모델 학습
stacking_model.fit(X_train, y_train)

# 예측
y_pred = stacking_model.predict(X_test)

# 평가
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
