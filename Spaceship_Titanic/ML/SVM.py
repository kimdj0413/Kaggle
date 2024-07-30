from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np

# 데이터 로드 및 전처리
df = pd.read_csv('D:/Kaggle/Spaceship_Titanic/Data/train_preprocess.csv')

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(df.drop('Transported', axis=1), df['Transported'], test_size=0.2, random_state=42)

# SVM 모델 정의
model = SVC(random_state=42)

# 그리드 서치에 사용할 파라미터 그리드 정의
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],          # 정규화 강도
    'kernel': ['linear', 'rbf', 'poly'],   # 커널 유형
    'gamma': ['scale', 'auto']             # 커널 계수
}

# 그리드 서치 정의
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           scoring='accuracy', cv=5, n_jobs=-1, verbose=2)

# 그리드 서치 실행
grid_search.fit(X_train, y_train)

# 최적 파라미터 및 성능 출력
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# 최적 모델로 예측 수행
best_model = grid_search.best_estimator_
pred = best_model.predict(X_test)

# 평가
accuracy = accuracy_score(y_test, pred)
conf_matrix = confusion_matrix(y_test, pred)
class_report = classification_report(y_test, pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
