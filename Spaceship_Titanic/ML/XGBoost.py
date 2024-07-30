from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
    
# 데이터 로드 및 전처리
df = pd.read_csv('D:/Kaggle/Spaceship_Titanic/Data/train_preprocess.csv')

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(df.drop('Transported', axis=1), df['Transported'], test_size=0.2, random_state=42)

# X_train.to_pickle('./Spaceship_Titanic/Data/X_train.pkl')
# X_test.to_pickle('./Spaceship_Titanic/Data/X_test.pkl')
# y_train.to_pickle('./Spaceship_Titanic/Data/y_train.pkl')
# y_test.to_pickle('./Spaceship_Titanic/Data/y_test.pkl')

# X_train = pd.read_pickle('./Spaceship_Titanic/Data/X_train.pkl')
# y_train = pd.read_pickle('./Spaceship_Titanic/Data/y_train.pkl')
# X_test = pd.read_pickle('./Spaceship_Titanic/Data/X_test.pkl')
# y_test = pd.read_pickle('./Spaceship_Titanic/Data/y_test.pkl')

# X_train = df.drop('Transported', axis=1)
# y_train = df['Transported']

# XGBoost 모델 및 GridSearchCV 설정
model = XGBClassifier()

# 하이퍼파라미터 그리드 설정
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    'max_depth': [3, 5, 7, 9, 11],
    'n_estimators': [100, 200, 300, 400, 500],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0],
    'min_child_weight': [1, 3, 5, 7],
    'scale_pos_weight': [1, 3, 5, 10]
    # 'max_depth': [5],
    # 'learning_rate': [0.05],
    # 'n_estimators': [300],
    # 'subsample': [1.0],
    # 'colsample_bytree': [0.6],
    # 'reg_lambda' : [0],
}

# GridSearchCV 객체 생성
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=10, verbose=2, n_jobs=-1)

# 모델 학습
grid_search.fit(X_train, y_train)

# 최적 하이퍼파라미터 출력
print("Best parameters found: ", grid_search.best_params_)

# 예측
pred = grid_search.predict(X_test)


# 평가
accuracy = accuracy_score(y_test, pred)
conf_matrix = confusion_matrix(y_test, pred)
class_report = classification_report(y_test, pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

print(pred)

X_test = pd.read_csv('D:/Kaggle/Spaceship_Titanic/Data/test_preprocess.csv')
pred = grid_search.predict(X_test)
print(pred)
pred_bool = [True if p == 1 else False for p in pred]
test_df = pd.read_csv('D:/Kaggle/Spaceship_Titanic/Data/test.csv')
test_df['Transported'] = pred_bool
result_df = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Transported': test_df['Transported']
})
result_df.to_csv('D:/Kaggle/Spaceship_Titanic/Data/result.csv', index=False)
print(test_df)
"""
# 모델 학습 후 특성 중요도 시각화
model = grid_search.best_estimator_  # GridSearchCV에서 최적 모델 가져오기

# 특성 중요도 계산
importance = model.feature_importances_

# 특성 이름
feature_names = X_train.columns

# 중요도를 데이터프레임으로 변환
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
"""