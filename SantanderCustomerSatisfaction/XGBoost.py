import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt

# 데이터프레임 불러오기 (예시로 CSV 파일을 사용)
df = pd.read_csv('D:/Kaggle/SantanderCustomerSatisfaction/Data/train_preprocess.csv')
df = df.drop(columns=['num_meses_var17_ult3', 'saldo_medio_var44_ult3', 'num_var20'])
# ID와 TARGET 열 제외
X = df.drop(columns=['ID','TARGET'])
y = df['TARGET']

# 트레인 테스트 셋 분리 (TARGET의 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 양성 클래스의 가중치 조정
scale_pos_weight = len(y_train) / (2 * np.sum(y_train))  # 0과 1의 비율에 따라 계산
print(scale_pos_weight)

# XGBoost 모델 생성
model = XGBClassifier(scale_pos_weight=35)

# 하이퍼파라미터 그리드 설정
param_grid = {
    'n_estimators': [100,200, 300, 400, 500],
    'max_depth': [7, 9, 11, 13],
    'learning_rate': [0.2, 0.3, 0.5, 0.7]
}

# 그리드 서치 설정
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# 모델 학습
grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터와 성능 출력
print("최적의 하이퍼파라미터:", grid_search.best_params_)
print("최고 정확도:", grid_search.best_score_)

# 테스트 셋 성능 평가
test_accuracy = grid_search.score(X_test, y_test)
print("테스트 셋 정확도:", test_accuracy)

# 예측 결과
y_pred_proba = grid_search.predict_proba(X_test)[:, 1]  # 양성 클래스에 대한 확률

# 원하는 임계값 설정 (예: 0.3)
threshold = 0.3

# 임계값에 따라 클래스 레이블 결정
y_pred_thresholded = (y_pred_proba >= threshold).astype(int)

# 혼동 행렬 및 클래스 리포트 출력
conf_matrix = confusion_matrix(y_test, y_pred_thresholded)
print("혼동 행렬:\n", conf_matrix)
print("클래스 리포트:\n", classification_report(y_test, y_pred_thresholded))

# AUC 스코어 계산
auc_score = roc_auc_score(y_test, y_pred_proba)
print("AUC 스코어:", auc_score)
"""
# 잔차 계산
residuals = y_test - y_pred_proba

# 잔차 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_proba, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted Probabilities')
plt.xlabel('Predicted Probabilities')
plt.ylabel('Residuals')
plt.show()

# 아웃라이어 찾기
# 잔차의 절대값이 특정 임계값(예: 0.1) 이상인 경우 아웃라이어로 간주
threshold = 0.1
outliers = np.where(np.abs(residuals) > threshold)[0]

# 아웃라이어 인덱스 및 수 출력
print("아웃라이어 인덱스:", outliers)
print("아웃라이어 수:", len(outliers))
outlierDf = df.drop(index=outliers)
outlierDf.to_csv('D:/Kaggle/SantanderCustomerSatisfaction/Data/train_outlier.csv', index=False)

# 피쳐 중요도 계산
importance = grid_search.best_estimator_.feature_importances_

# 피쳐 이름과 중요도를 데이터프레임으로 변환
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})

# 중요도를 백분율로 변환
importance_df['Importance'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100

# 중요도 백분율로 정렬
importance_df = importance_df.sort_values(by='Importance', ascending=False)
# 중요도가 2% 미만인 피쳐 이름 추출
low_importance_features = importance_df[importance_df['Importance'] < 0.7]['Feature'].tolist()

# 결과 출력
print("중요도가 2% 미만인 컬럼들:", low_importance_features)
# 피쳐 중요도 시각화
plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance (%)')
plt.title('Feature Importance (Percentage)')
plt.gca().invert_yaxis()  # 중요도가 높은 순서대로 표시
plt.show()
"""
# 새로운 CSV 파일 불러오기
new_data = pd.read_csv('D:/Kaggle/SantanderCustomerSatisfaction/Data/test_preprocess.csv')  # 예시 파일 경로
id_data = pd.read_csv('D:/Kaggle/SantanderCustomerSatisfaction/Data/test.csv')  # 예시 파일 경로
new_data = new_data.drop(columns=['num_meses_var17_ult3', 'saldo_medio_var44_ult3', 'num_var20'])

# ID 열 추출
id_column = id_data['ID']

# 새로운 데이터에서 ID 열 제외하고 예측에 사용할 피처만 추출
X_new = new_data.drop(columns=['ID'])

# 예측 확률 계산
y_new_pred_proba = grid_search.predict_proba(X_new)[:, 1]  # 양성 클래스에 대한 확률

# 원하는 임계값 설정 (예: 0.3)
threshold = 0.3

# 임계값에 따라 클래스 레이블 결정
y_new_pred = (y_new_pred_proba >= threshold).astype(int)

# 예측 결과를 새로운 데이터프레임으로 생성
result_df = pd.DataFrame({'ID': id_column, 'TARGET': y_new_pred})

# 결과를 result.csv로 저장
result_df.to_csv('D:/Kaggle/SantanderCustomerSatisfaction/Data/result.csv', index=False)

print("예측 결과가 result.csv에 저장되었습니다.")
