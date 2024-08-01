import pandas as pd

# 데이터 불러오기
df = pd.read_csv('D:/Kaggle/Titanic/Data/result.csv')


original_survived = df['Survived'].copy()
condition = df['Survival Probability'] < 0.03
df.loc[condition, 'Survived'] = df.loc[condition, 'Survived'].apply(lambda x: 1 if x == 0 else 0)
df.drop(columns=['Survival Probability'], inplace=True)
df.to_csv('D:/Kaggle/Titanic/Data/proba_result.csv', index=False)
changed_count = (original_survived != df['Survived']).sum()
total_count = len(df)
change_ratio = changed_count / total_count
print(f"변경된 Survived 열의 비율: {change_ratio:.4f}")