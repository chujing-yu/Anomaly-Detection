from methods import *

# train_new = pd.read_csv('dataset/WADI.A2_19 Nov 2019/WADI_14days_new.csv')
# test_new = pd.read_csv('dataset/WADI.A2_19 Nov 2019/WADI_attackdataLABLE.csv', skiprows=1)
#
# test = pd.read_csv('dataset/WADI.A1_9 Oct 2017/WADI_attackdata.csv')
# train = pd.read_csv('dataset/WADI.A1_9 Oct 2017/WADI_14days.csv', skiprows=4)
#
#
# def recover_date(str1, str2):
#     return str1 + " " + str2
#
#
# train["datetime"] = train.apply(lambda x: recover_date(x['Date'], x['Time']), axis=1)
# train["datetime"] = pd.to_datetime(train['datetime'])
#
# train_time = train[['Row', 'datetime']]
# train_new_time = pd.merge(train_new, train_time, how='left', on='Row')
# del train_new_time['Row']
# del train_new_time['Date']
# del train_new_time['Time']
# train_new_time.to_csv('./processed/WADI_train.csv', index=False)
#
# test["datetime"] = test.apply(lambda x: recover_date(x['Date'], x['Time']), axis=1)
# test["datetime"] = pd.to_datetime(test['datetime'])
# test = test.loc[-2:, :]
# test_new = test_new.rename(columns={'Row ': 'Row'})
#
# test_time = test[['Row', 'datetime']]
# test_new_time = pd.merge(test_new, test_time, how='left', on='Row')
#
# del test_new_time['Row']
# del test_new_time['Date ']
# del test_new_time['Time']
#
# test_new_time = test_new_time.rename(columns={'Attack LABLE (1:No Attack, -1:Attack)': 'label'})
# test_new_time.loc[test_new_time['label'] == 1, 'label'] = 0
# test_new_time.loc[test_new_time['label'] == -1, 'label'] = 1
#
# test_new_time.to_csv('./processed/WADI_test.csv', index=False)


train_data = pd.read_csv('dataset/WADI.A2_19 Nov 2019/WADI_14days_new.csv')
test_data = pd.read_csv('dataset/WADI.A2_19 Nov 2019/WADI_attackdataLABLE.csv', skiprows=1)

# 这几列都是NaN值，直接赋值0
ncolumns = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS']
train_data[ncolumns] = 0
test_data[ncolumns] = 0

test_data.rename(columns={'Attack LABLE (1:No Attack, -1:Attack)': 'label'}, inplace=True)
test_data.loc[test_data['label'] == 1, 'label'] = 0
test_data.loc[test_data['label'] == -1, 'label'] = 1


# window = 10
# # 相当于滑动窗口10， stride1的结果，取每个index为10的倍数就是window10，stride10
# train_data_mean = train_data[:, 3:].rolling(window).mean()
# train_data_mean = train_data_mean[(train_data_mean.index + 1) % window == 0]
# test_data_mean = test_data[:, 3:-1].rolling(window).mean()
# test_data_mean = test_data_mean[(test_data_mean.index + 1) % window == 0]

# 还有一些NaN数值就用上一条数据填充
train_data_mean = train_data.fillna(method='ffill')
test_data_mean = test_data.fillna(method='ffill')

# 取127个特征
x_train = train_data_mean.iloc[:, 3:]
x_test = test_data_mean.iloc[:, 3:-1]

# 标签设置为窗口内的多数，由于取平均，假设10条里大于5条的设置为异常
# f = lambda s: 1 if s > 0.5 else 0
# y_test = test_data_mean['label'].apply(f)
y_test = test_data_mean['label']

# 最大最小值归一化
scaler = MinMaxScaler()  # 实例化
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

print(x_train.shape)

pd.DataFrame(x_train).to_csv('processed/wadi/x_train.csv', index=False)

pd.DataFrame(x_test).to_csv('processed/wadi/x_test.csv', index=False)
pd.DataFrame(y_test).to_csv('processed/wadi/y_test.csv', index=False)










# normalized_train = train
# normalized_test = test
# scaler = MinMaxScaler(feature_range=(0, 1))
# for col in train.columns:
#     print(col)
#     if len(train[col].unique()) == 1:
#         # 如果该列只有一个唯一值，将其直接添加到归一化后的数据中
#         normalized_train[col] = train[col]
#     else:
#         # 对其他列进行归一化处理
#         scaled_col = scaler.fit_transform(train[col].values.reshape(-1, 1))
#         normalized_train[col] = scaled_col
#
#
# for col in train.columns:
#     if len(test[col].unique()) == 1:
#         # 如果该列只有一个唯一值，将其直接添加到归一化后的数据中
#         normalized_test[col] = test[col]
#     else:
#         # 对其他列进行归一化处理
#         scaled_col = scaler.fit_transform(test[col].values.reshape(-1, 1))
#         normalized_test[col] = scaled_col
#
# offset = 0.001
# for col in train.columns:
#     # 使用列表推导对所有的0值添加偏移量
#     normalized_train[col] = [value + offset if value == 0.0 else value for value in normalized_train[col]]
#
#
# normalized_train.to_csv('processed/wadi/x_train.csv', index=False)
#
# x_test = normalized_test.drop('label', axis=1)
# y_test = normalized_test['label']
# x_test.to_csv('processed/wadi/x_test.csv', index=False)
# y_test.to_csv('processed/wadi/y_test.csv', index=False)
