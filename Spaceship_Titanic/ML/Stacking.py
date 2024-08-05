import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#####################################
##      데이터 불러오기 및 처리     ##
#####################################
df = pd.read_csv('D:/Kaggle/Spaceship_Titanic/Data/train_preprocess2.csv')
X = df.drop('Transported', axis=1)
y = df['Transported']
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

#############################################
##      모델 및 하이퍼 파라미터 설정        ##
#############################################
# 개별 모델 설정
xgb_model = XGBClassifier(scale_pos_weight=0.76, eval_metric='logloss', n_estimators=30, max_depth=3, 
                           subsample=0.8, reg_lambda=2, gamma=0.5)

# 랜덤 포레스트 모델과 하이퍼파라미터 그리드 설정
rf_model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50],
    'criterion': ['gini', 'entropy']
}

# GridSearchCV를 사용하여 랜덤 포레스트 하이퍼파라미터 튜닝
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(XTrain, yTrain)

# 최적의 랜덤 포레스트 모델
best_rf_model = grid_search.best_estimator_
print("Best Random Forest Parameters: ", grid_search.best_params_)

# 스태킹 모델 설정
estimators = [
    ('xgb', xgb_model),
    ('rf', best_rf_model)  # 최적의 랜덤 포레스트 모델 사용
]

stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

######################
##      학습        ##
######################
stacking_model.fit(XTrain, yTrain)

yPred = stacking_model.predict(XTest)

######################
##      평가        ##
######################
accuracy = accuracy_score(yTest, yPred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(yTest, yPred))
print("Confusion Matrix:")
print(confusion_matrix(yTest, yPred))

XNew = pd.read_csv('D:/Kaggle/Spaceship_Titanic/Data/test_preprocess2.csv')
tempDf = pd.read_csv('D:/Kaggle/Spaceship_Titanic/Data/test.csv')
passengerId = tempDf['PassengerId']
yNewPred = stacking_model.predict(XNew)
resultDf = pd.DataFrame({'PassengerId':passengerId, 'Transported':yNewPred})
resultDf['Transported'] = resultDf['Transported'].map({1: True, 0: False})
resultDf.to_csv('D:/Kaggle/Spaceship_Titanic/Data/result.csv', index=False)