import pandas as pd

df = pd.read_csv('D:/Kaggle/Titanic/Data/train_preprocess.csv')
print(len(df))
outlierCol = [709, 137, 621, 447, 192, 673, 396, 141, 235, 204, 23, 659, 44, 772, 312, 889, 767, 357, 254, 174, 165, 625, 712, 338, 286, 209, 512, 527, 78, 604, 139, 65, 599, 830, 49, 854, 109, 643, 657, 507, 54, 97, 572, 852, 25]
print(len(outlierCol))
for col in outlierCol:
  df = df.drop(index=col)
df.to_csv('D:/Kaggle/Titanic/Data/train_preprocess_outlier.csv', index=False)
print(len(df))

print(df['Survived'].value_counts(normalize=True))