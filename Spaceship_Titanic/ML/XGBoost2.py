import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
model = XGBClassifier(scale_pos_weight=0.77,eval_metric='logloss')
#0.71
param_grid = {
  'subsample': [0.8],
  'n_estimators': [30],
  'max_depth': [3],
  'lambda': [2],
  'gamma': [0.5]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           scoring='accuracy', cv=10, verbose=1, n_jobs=-1)

######################
##      학습        ##
######################
grid_search.fit(XTrain, yTrain)
print("Best parameters found: ", grid_search.best_params_)

yPred = grid_search.predict(XTest)
######################
##      평가        ##
######################
accuracy = accuracy_score(yTest, yPred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(yTest, yPred))
print("Confusion Matrix:")
print(confusion_matrix(yTest, yPred))

"""
#################################
##      이상치 분석(잔차)       ##
#################################
# 잔차 계산
yPredProba = grid_search.predict_proba(XTest)[:, 1]
residuals = yTest - yPredProba

threshold = 0.43 #      잔차 임계값
outliers = XTest[np.abs(residuals) > threshold]
print(len(outliers))

# 이상치의 인덱스
outlier_indices = XTest.index[np.abs(residuals) > threshold]
print("잔차가 큰 이상치의 인덱스:")
print(outlier_indices.tolist())
print(len(outlier_indices)/len(df)*100)
#0.19

#################################
##      피처 중요도 시각화      ##
#################################
importance = grid_search.best_estimator_.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False) #       중요도에 따라 정렬

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()
"""

##################################
##      test 데이터 예측        ##
#################################
XNew = pd.read_csv('D:/Kaggle/Spaceship_Titanic/Data/test_preprocess2.csv')
tempDf = pd.read_csv('D:/Kaggle/Spaceship_Titanic/Data/test.csv')
passengerId = tempDf['PassengerId']
yNewPred = grid_search.predict(XNew)
resultDf = pd.DataFrame({'PassengerId':passengerId, 'Transported':yNewPred})
resultDf['Transported'] = resultDf['Transported'].map({1: True, 0: False})
resultDf.to_csv('D:/Kaggle/Spaceship_Titanic/Data/result.csv', index=False)
