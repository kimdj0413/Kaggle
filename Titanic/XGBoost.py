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
df = pd.read_csv('D:/Kaggle/Titanic/Data/train_preprocess_outlier.csv')

X = df.drop('Survived', axis=1)
y = df['Survived']
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

#########################################
##      불균형 데이터 가중치 설정       ##
#########################################
posCnt = sum(yTrain)
negCnt = len(yTrain) - posCnt
scalePosWeight = posCnt / negCnt

#############################################
##      모델 및 하이퍼 파라미터 설정        ##
#############################################
model = XGBClassifier(scale_pos_weight=scalePosWeight,eval_metric='logloss')

param_grid = {
    'colsample_bytree': [0.3],
    'gamma': [0.6],
    'min_child_weight': [1],
    'n_estimators': [50]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           scoring='accuracy', cv=100, verbose=1, n_jobs=-1)

######################
##      학습        ##
######################
grid_search.fit(XTrain, yTrain)
print("Best parameters found: ", grid_search.best_params_)

#################################
##      예측 임계값 조정        ##
#################################
probaRate = 0.3
yPredProba = grid_search.predict_proba(XTest)[:, 1] #       양성 클래스 확률

mean_proba = np.mean(yPredProba)

threshold = probaRate
yPred = (yPredProba >= threshold).astype(int)

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
residuals = yTest - yPredProba

threshold = 0.15 #      잔차 임계값
outliers = XTest[np.abs(residuals) > threshold]
print(len(outliers))

# 이상치의 인덱스
outlier_indices = XTest.index[np.abs(residuals) > threshold]
print("잔차가 큰 이상치의 인덱스:")
print(outlier_indices.tolist())
"""
"""
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
XNew = pd.read_csv('D:/Kaggle/Titanic/Data/test_preprocess.csv')
tempDf = pd.read_csv('D:/Kaggle/Titanic/Data/test.csv')
passengerId = tempDf['PassengerId']
yNewPred_proba = grid_search.predict(XNew)
yNewPred_proba = grid_search.predict_proba(XNew)[:, 1]
threshold = probaRate
yNewPred = (yNewPred_proba >= threshold).astype(int)
resultDf = pd.DataFrame({'PassengerId':passengerId, 'Survived':yNewPred})
resultDf.to_csv('D:/Kaggle/Titanic/Data/result.csv', index=False)
