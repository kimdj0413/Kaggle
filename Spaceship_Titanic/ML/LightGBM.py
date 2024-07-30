import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드 및 전처리
df = pd.read_csv('D:/Kaggle/Spaceship_Titanic/Data/train_preprocess.csv')
print(df)

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(df.drop('Transported', axis=1), df['Transported'], test_size=0.2, random_state=42)

# LightGBM 모델 및 GridSearchCV 설정
model = lgb.LGBMClassifier()

# 하이퍼파라미터 그리드 설정
param_grid = {
    # 'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    # 'max_depth': [3, 5, 7, 9, 11],
    # 'n_estimators': [100, 200, 300, 400, 500],
    # 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # 'colsample_bytree': [0.6, 0.8, 1.0]
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
    # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # 'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    # 'max_depth': [-1, 3, 5, 7, 9, 11],  # -1은 무제한 깊이를 의미
    # 'n_estimators': [100, 200, 300, 400, 500],
    # 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

# GridSearchCV 객체 생성
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=2, n_jobs=-1)

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
