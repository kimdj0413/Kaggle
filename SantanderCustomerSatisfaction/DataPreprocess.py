import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import CondensedNearestNeighbour, ClusterCentroids

######################
##    데이터 파악   ##
######################
df = pd.read_csv('D:/Kaggle/SantanderCustomerSatisfaction/Data/train.csv') #_preprocess_temp
beforeCol = df.columns
################################
##    모든 행이 0인 열 삭제   ##
################################
zero_columns = [col for col in df.columns if (df[col] == 0).all()]
for col in zero_columns:
  df = df.drop(columns=[col])

########################################
##    상관관계 0.85 이상인 열 삭제    ##
########################################
correlation_matrix = df.corr()
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.85:
          if(correlation_matrix.columns[i] != correlation_matrix.columns[j] and correlation_matrix.columns[i] != 'ID' and correlation_matrix.columns[j] != 'ID' and correlation_matrix.columns[i] != 'TARGET' and correlation_matrix.columns[j] != 'TARGET'):
            try:
              df = df.drop(columns=[correlation_matrix.columns[j]])
            except KeyError as e:
              pass

##############################################
##    상관관계 0.85 >= x > 0.7 인 열 병합   ##
##############################################
correlation_matrix = df.corr()
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
          if(correlation_matrix.columns[i] != correlation_matrix.columns[j] and correlation_matrix.columns[i] != 'ID' and correlation_matrix.columns[j] != 'ID' and correlation_matrix.columns[i] != 'TARGET' and correlation_matrix.columns[j] != 'TARGET'):
            mainCol = correlation_matrix.columns[i]
            subCol = correlation_matrix.columns[j]
            try:
              df[mainCol] = df[mainCol] + df[subCol]
              df = df.drop(columns=[subCol])
            except KeyError as e:
              pass

##########################################
##    상관관계 0.0001 미만인 열 병합    ##
##########################################
correlation_matrix = df.corr()
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) < 0.0001:
          if(correlation_matrix.columns[i] != correlation_matrix.columns[j] and correlation_matrix.columns[i] != 'ID' and correlation_matrix.columns[j] != 'ID' and correlation_matrix.columns[i] != 'TARGET' and correlation_matrix.columns[j] != 'TARGET'): # and correlation_matrix.columns[i] != 'ID' and correlation_matrix.columns[j] != 'ID'
            high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))
            mainCol = correlation_matrix.columns[i]
            subCol = correlation_matrix.columns[j]
            try:
              df[mainCol] = df[mainCol] + df[subCol]
              df = df.drop(columns=[subCol])
            except KeyError as e:
              pass

############################
##    로버스트 스케일링   ##
############################
scaler = RobustScaler()
features = df.drop(columns=['TARGET'])
scaledFeatures = scaler.fit_transform(features)
scaledDf = pd.DataFrame(scaledFeatures, columns=features.columns)
df = pd.concat([scaledDf, df['TARGET']], axis=1)

df.to_csv('D:/Kaggle/SantanderCustomerSatisfaction/Data/train_preprocess.csv', index=False)

"""
X = df.drop(columns=['ID', 'TARGET'])
y = df['TARGET']

# 비율 확인
print("원본 TARGET 비율:")
print(y.value_counts())

# Cluster Centroids 언더샘플링 적용
cc = ClusterCentroids(random_state=42)
X_resampled, y_resampled = cc.fit_resample(X, y)

# 언더샘플링 후 비율 확인
print("언더샘플링 후 TARGET 비율:")
print(y_resampled.value_counts())

# 언더샘플링된 데이터프레임 생성
resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df['TARGET'] = y_resampled

# 언더샘플링된 데이터프레임 확인
print("언더샘플링된 데이터프레임 크기:", resampled_df.shape)

# 언더샘플링된 데이터프레임을 CSV 파일로 저장
resampled_df.to_csv('D:/Kaggle/SantanderCustomerSatisfaction/Data/train_preprocess.csv', index=False)

print("언더샘플링된 데이터프레임이 'resampled_data.csv'로 저장되었습니다.")

############################
##    ADASYN 오버샘플링   ##
############################

# X = df.drop('TARGET', axis=1)
# y = df['TARGET']
# ada = ADASYN(sampling_strategy={1:4000, 0:4000}, random_state=42)
# XSampling, ySampling = ada.fit_resample(X, y)
# df = pd.DataFrame(XSampling, columns=X.columns)
# df['TARGET'] = ySampling

print(df['TARGET'].value_counts())
# afterCol = df.columns
# minusCol = set(beforeCol) - set(afterCol)

# print(f'전처리 전 컬럼 : {beforeCol} ==> {len(beforeCol)}개')
# print(f'전처리 후 컬럼 : {afterCol} ==> {len(afterCol)}개')
# print(f'뺀 컬럼 : {minusCol} ==> {len(minusCol)}개')

# df.to_csv('D:/Kaggle/SantanderCustomerSatisfaction/Data/train_preprocess.csv', index=False)

######################
##    테스트 처리   ##
######################
df = pd.read_csv('D:/Kaggle/SantanderCustomerSatisfaction/Data/test.csv')
df = df.drop(columns=['num_trasp_var17_out_hace3', 'num_op_var39_hace3', 'saldo_var26', 'num_meses_var12_ult3', 'ind_var28', 'ind_var26_0', 'ind_var46_0', 'ind_var10_ult1', 'num_var1', 'num_trasp_var17_out_ult1', 'num_aport_var33_hace3', 'imp_op_var40_comer_ult3', 'ind_var40_0', 'delta_imp_venta_var44_1y3', 'ind_var39_0', 'delta_imp_amort_var34_1y3', 'saldo_var32', 'ind_var5', 'imp_compra_var44_hace3', 'num_op_var41_ult1', 'num_op_var40_ult1', 'saldo_var17', 'num_reemb_var17_hace3', 'imp_aport_var13_hace3', 'ind_var25_cte', 'num_var27_0', 'saldo_var30', 'num_op_var40_ult3', 'ind_var33', 'saldo_medio_var13_medio_hace2', 'imp_aport_var13_ult1', 'num_var39_0', 'ind_var18_0', 'delta_num_venta_var44_1y3', 'num_var30', 'num_var1_0', 'num_var31_0', 'ind_var8', 'saldo_medio_var13_largo_hace2', 'ind_var25_0', 'delta_imp_compra_var44_1y3', 'imp_op_var39_efect_ult3', 'imp_aport_var33_hace3', 'num_var8', 'ind_var13_largo_0', 'imp_op_var41_efect_ult3', 'saldo_medio_var44_hace2', 'ind_var6', 'imp_op_var39_comer_ult3', 'num_var4', 'ind_var29', 'imp_trasp_var33_out_hace3', 'num_var32_0', 'num_trasp_var33_out_ult1', 'num_med_var45_ult3', 'ind_var12', 'num_aport_var17_hace3', 'ind_var13_medio_0', 'num_var20_0', 'ind_var37_cte', 'num_var12_0', 'num_var37_med_ult2', 'num_op_var39_ult1', 'imp_var7_emit_ult1', 'num_var41_0', 'num_trasp_var17_in_ult1', 'saldo_var24', 'imp_reemb_var13_hace3', 'saldo_var42', 'ind_var7_recib_ult1', 'num_op_var39_hace2', 'ind_var12_0', 'saldo_medio_var44_ult1', 'num_var13_medio_0', 'saldo_medio_var13_corto_hace2', 
'num_var46', 'num_var6', 'saldo_var6', 'delta_imp_amort_var18_1y3', 'num_var5', 'ind_var32_0', 'saldo_var33', 'num_var26_0', 'num_reemb_var33_hace3', 'num_op_var41_hace3', 'num_meses_var29_ult3', 'num_var26', 'imp_reemb_var17_ult1', 'num_op_var40_hace3', 'saldo_medio_var13_corto_ult1', 'saldo_medio_var33_ult1', 'num_var30_0', 'saldo_medio_var8_hace2', 'delta_imp_aport_var17_1y3', 'num_op_var39_comer_ult3', 'num_trasp_var33_in_hace3', 'ind_var29_0', 'num_meses_var13_corto_ult3', 'ind_var44', 'saldo_var28', 'ind_var17', 'num_var13_corto', 'num_var45_hace2', 'num_var25', 'imp_op_var40_efect_ult1', 'num_op_var40_efect_ult3', 'ind_var13_medio', 'num_var29_0', 'ind_var41', 'ind_var26', 'num_meses_var8_ult3', 'imp_op_var40_ult1', 'num_var45_hace3', 'num_var28_0', 'delta_num_reemb_var17_1y3', 'delta_num_trasp_var33_in_1y3', 'ind_var34_0', 'imp_var43_emit_ult1', 'saldo_var13_medio', 'num_op_var40_comer_ult1', 'num_reemb_var17_ult1', 'ind_var37', 'delta_imp_trasp_var17_out_1y3', 'num_op_var40_efect_ult1', 'num_op_var39_efect_ult1', 'imp_var7_recib_ult1', 'imp_venta_var44_ult1', 'ind_var25', 'ind_var1_0', 'num_trasp_var33_out_hace3', 'imp_venta_var44_hace3', 'ind_var2', 'imp_op_var39_efect_ult1', 'imp_op_var39_comer_ult1', 'saldo_var14', 'ind_var30', 'saldo_var2_ult1', 'imp_amort_var34_ult1', 'ind_var46', 'num_var8_0', 'ind_var27', 'num_op_var41_comer_ult1', 'imp_ent_var16_ult1', 'imp_op_var39_ult1', 'saldo_medio_var17_hace3', 'delta_imp_reemb_var13_1y3', 'saldo_var8', 'num_var34_0', 'ind_var18', 'saldo_medio_var13_medio_ult1', 'ind_var39', 'num_meses_var44_ult3', 'saldo_var12', 'saldo_var29', 'imp_trasp_var17_in_hace3', 'num_venta_var44_ult1', 'num_var22_hace2', 'saldo_var25', 'ind_var32', 'delta_num_aport_var13_1y3', 'num_var13_largo_0', 'num_compra_var44_hace3', 'imp_op_var41_comer_ult1', 'num_var13_largo', 'num_var13_corto_0', 'num_trasp_var33_in_ult1', 'saldo_medio_var13_medio_hace3', 'ind_var17_0', 'ind_var1', 'num_var40', 'saldo_medio_var29_ult1', 'delta_imp_trasp_var17_in_1y3', 'num_op_var39_comer_ult1', 'num_reemb_var13_ult1', 'imp_aport_var17_hace3', 'saldo_medio_var13_largo_ult1', 'saldo_medio_var13_corto_ult3', 'num_var45_ult1', 'num_var18_0', 'saldo_var40', 'num_var17', 'num_var35', 'saldo_var13_largo', 'num_var2_0_ult1', 'ind_var26_cte', 'saldo_var20', 'saldo_medio_var12_hace2', 'var3', 'num_var43_emit_ult1', 'num_var6_0', 'ind_var13', 'num_var13_medio', 'imp_op_var41_efect_ult1', 'ind_var32_cte', 'ind_var13_corto', 'num_op_var41_ult3', 'saldo_var5', 'imp_op_var40_comer_ult1', 'saldo_medio_var5_ult1', 'num_var2_ult1', 'num_venta_var44_hace3', 'delta_num_reemb_var33_1y3', 'num_var22_ult3', 'saldo_var18', 'ind_var8_0', 'ind_var20_0', 'num_var46_0', 'delta_imp_aport_var33_1y3', 'ind_var44_0', 'imp_trans_var37_ult1', 'saldo_var44', 'num_meses_var13_medio_ult3', 'ind_var14_0', 'imp_reemb_var17_hace3', 'imp_compra_var44_ult1', 'num_op_var41_efect_ult1', 'num_meses_var13_largo_ult3', 'num_var24', 'imp_trasp_var33_in_ult1', 'num_var33_0', 'saldo_medio_var29_ult3', 'saldo_medio_var5_hace2', 'imp_trasp_var33_in_hace3', 'saldo_medio_var17_ult1', 'num_op_var39_ult3', 'imp_trasp_var17_out_ult1', 'imp_trasp_var33_out_ult1', 'imp_aport_var17_ult1', 'num_var41', 'num_var42', 'ind_var20', 'ind_var33_0', 'num_trasp_var11_ult1', 'num_var18', 'imp_op_var40_efect_ult3', 'ind_var13_largo', 'imp_op_var41_ult1', 'ind_var37_0', 'ind_var6_0', 'num_var28', 'num_var32', 'saldo_medio_var33_hace2', 'ind_var28_0', 'delta_imp_aport_var13_1y3', 'delta_num_compra_var44_1y3', 'num_var43_recib_ult1', 'imp_reemb_var13_ult1', 'saldo_var27', 'num_var12', 'num_op_var41_efect_ult3', 'ind_var9_cte_ult1', 'ind_var34', 'saldo_medio_var29_hace3', 'saldo_var46', 'saldo_medio_var17_hace2', 'delta_num_trasp_var17_in_1y3', 'num_var29', 'delta_num_aport_var17_1y3', 'imp_amort_var18_hace3', 'delta_num_reemb_var13_1y3', 'num_var37_0', 'num_trasp_var17_in_hace3', 'ind_var24', 'num_var40_0', 'num_var17_0', 'delta_imp_trasp_var33_in_1y3', 'ind_var10cte_ult1', 'saldo_var34', 'delta_imp_trasp_var33_out_1y3', 'num_var44_0', 'saldo_medio_var8_hace3', 'ind_var13_corto_0', 'ind_var2_0', 'imp_reemb_var33_ult1', 'saldo_medio_var17_ult3', 'imp_amort_var34_hace3', 'delta_num_aport_var33_1y3', 'num_aport_var17_ult1', 'saldo_medio_var29_hace2', 'num_var13_0', 'imp_trasp_var17_out_hace3', 'num_var24_0', 'num_reemb_var33_ult1', 'ind_var31', 'ind_var31_0', 'delta_num_trasp_var17_out_1y3', 'num_var25_0', 'saldo_var41', 'num_compra_var44_ult1', 'num_meses_var33_ult3', 'num_var33', 'imp_amort_var18_ult1', 'ind_var14', 'ind_var40', 'ind_var41_0', 'saldo_var1', 'imp_op_var41_comer_ult3', 'delta_num_trasp_var33_out_1y3', 'num_var44', 'delta_imp_reemb_var17_1y3', 'saldo_medio_var8_ult1', 'ind_var13_0', 'num_op_var41_hace2', 'imp_trasp_var17_in_ult1', 'imp_sal_var16_ult1', 'delta_imp_reemb_var33_1y3', 'num_var27', 'saldo_var13_corto', 'num_var34', 'saldo_medio_var12_ult1', 'ind_var24_0', 'imp_aport_var33_ult1', 'num_var13', 'num_op_var40_hace2', 'saldo_var31', 'saldo_medio_var5_ult3', 'imp_reemb_var33_hace3', 'num_var7_emit_ult1', 'ind_var5_0', 'saldo_medio_var13_medio_ult3', 'saldo_var13', 'ind_var27_0', 'num_reemb_var13_hace3', 'num_aport_var33_ult1', 'ind_var7_emit_ult1'])
scaler = RobustScaler()
scaledFeatures = scaler.fit_transform(df)
scaledDf = pd.DataFrame(scaledFeatures, columns=df.columns)
scaledDf.to_csv('D:/Kaggle/SantanderCustomerSatisfaction/Data/test_preprocess.csv', index=False)
"""