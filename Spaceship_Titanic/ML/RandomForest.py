from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# 데이터 로드 및 전처리
df = pd.read_csv('D:/Kaggle/Spaceship_Titanic/Data/train_preprocess.csv')

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(df.drop('Transported', axis=1), df['Transported'], test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 정의 (파라미터를 설정하여 직접 정의)
model = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=5, 
                               min_samples_leaf=1, max_features='log2', random_state=42)

# 모델 학습
model.fit(X_train, y_train)

# 예측 수행
pred = model.predict(X_test)

# 평가
accuracy = accuracy_score(y_test, pred)
conf_matrix = confusion_matrix(y_test, pred)
class_report = classification_report(y_test, pred)

# 결과 출력
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
