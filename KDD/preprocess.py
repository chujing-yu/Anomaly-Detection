from methods import *

# Load dataset
train_file = 'dataset/KDDTrain+.txt'
train_data = pd.read_csv(train_file, header=None)
test_file = 'dataset/KDDTest+.txt'
test_data = pd.read_csv(test_file, header=None)

# 添加列名
column_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label',
                'Difficulty Level']

train_data.columns = column_names
test_data.columns = column_names

# 将标签列进行二分类处理，将正常流量标记为0，将攻击流量标记为1
train_data['label'] = train_data['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_data['label'] = test_data['label'].apply(lambda x: 0 if x == 'normal' else 1)

sympolic_columns = ["protocol_type", "service", "flag"]
label_column = "label"
# df = pd.DataFrame()
for column in column_names:
    if column in sympolic_columns:
        encode_text(train_data, test_data, column)
    elif not column == label_column:
        minmax_scale_values(train_data, test_data, column)

# Train

x_train = train_data.drop('label', axis=1)
y_train = train_data['label']
# 125973 123
x_train.to_csv('dataset/x_train.csv', index=False)
y_train.to_csv('dataset/y_train.csv', index=False)

x_test = test_data.drop('label', axis=1)
y_test = test_data['label']
x_test.to_csv('dataset/x_test.csv', index=False)
y_test.to_csv('dataset/y_test.csv', index=False)
