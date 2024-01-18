import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import recall_score, precision_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# 加载WADI数据集
# Load training dataset
from tensorflow.python.keras.metrics import accuracy

train_file = 'G:/Dataset/WADI/WADI.A2_19 Nov 2019/WADI_14days_new.csv'
train_data = pd.read_csv(train_file)

# Load testing dataset
test_file = 'G:/Dataset/WADI/WADI.A2_19 Nov 2019/WADI_attackdataLABLE.csv'
test_data = pd.read_csv(test_file, skiprows=1)

# 这几列都是NaN值，直接赋值0
ncolumns = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS']
train_data[ncolumns] = 0
test_data[ncolumns] = 0

# 标签列1为异常，-1为正常 修改为 1为异常，0为正常
test_data.rename(columns={'Attack LABLE (1:No Attack, -1:Attack)': 'label'}, inplace=True)
test_data.loc[test_data['label'] == 1, 'label'] = 0
test_data.loc[test_data['label'] == -1, 'label'] = 1

# 相当于滑动窗口10， stride1的结果，取每个index为10的倍数就是window10，stride10
train_data_mean = train_data.rolling(10).mean()
train_data_mean = train_data_mean[(train_data_mean.index + 1) % 10 == 0]
test_data_mean = test_data.rolling(10).mean()
test_data_mean = test_data_mean[(test_data_mean.index + 1) % 10 == 0]

# 还有一些NaN数值就用上一条数据填充
train_data_mean = train_data_mean.fillna(method='ffill')
test_data_mean = test_data_mean.fillna(method='ffill')

# 取127个特征
x_train = train_data_mean.iloc[:, 1:].values
x_test = test_data_mean.iloc[:, 1:-1].values

# 标签设置为窗口内的多数，由于取平均，假设10条里大于5条的设置为异常
f = lambda s: 1 if s > 0.5 else 0

y_test = test_data_mean['label'].apply(f)

# 最大最小值归一化
scaler = MinMaxScaler()  # 实例化
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# 构建VAE模型
original_dim = x_train.shape[1]
latent_dim = 2
intermediate_dim = 512

encoder_inputs = keras.Input(shape=(original_dim,))
h = layers.Dense(intermediate_dim, activation='relu')(encoder_inputs)
h = layers.Dense(intermediate_dim, activation='relu')(h)
# 计算p(Z|X)的均值和方差
z_mean = layers.Dense(latent_dim)(h)
z_log_var = layers.Dense(latent_dim)(h)


# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# 重参数层，相当于给输入加入噪声
z = layers.Lambda(sampling,output_shape=(latent_dim,))([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
# 解码层
"""decoder_outputs = layers.Dense(train_data.shape[1], activation='sigmoid')(x)

decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')

vae = keras.Model(encoder_inputs, decoder(encoder(encoder_inputs)[2]), name='vae')"""
"""decoder_h = layers.Dense(intermediate_dim,activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))
decoder_mean = layers.Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)"""

latent_inputs = layers.Input(shape=(latent_dim,))
h = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
h = layers.Dense(intermediate_dim, activation='relu')(h)

decoder_outputs = layers.Dense(original_dim, activation='sigmoid')(h)

decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')

# 建立模型
# vae = keras.Model(encoder_inputs, x_decoded_mean)
vae_outputs = decoder(encoder(encoder_inputs)[2])
vae =keras.Model(encoder_inputs, vae_outputs, name='vae')

# 定义VAE的损失函数
"""reconstruction_loss = keras.backend.sum(keras.backend.binary_crossentropy(encoder_inputs, x_decoded_mean), axis=-1)
kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
kl_loss = tf.reduce_mean(kl_loss) * -0.5

vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)"""

reconstruction_loss = keras.backend.sum(keras.backend.binary_crossentropy(encoder_inputs, vae_outputs), axis=-1)
kl_loss = -0.5 * keras.backend.sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)

# 编译并训练VAE模型
vae.compile(optimizer='adam')
vae.fit(x_train,
        shuffle=True,
        epochs=20,
        batch_size=100,
        validation_data=(x_test, None))

# 使用训练好的VAE模型进行异常检测
reconstructed_data = vae.predict(x_test)
mse = np.mean(np.power(x_test - reconstructed_data, 2), axis=1)
threshold = np.percentile(mse, 98)  # 设置异常检测的阈值

testing_set_predictions = vae.predict(x_test)
test_losses = mse
testing_set_predictions = np.zeros(len(test_losses))
testing_set_predictions[np.where(test_losses > threshold)] = 1

recall = recall_score(y_test, testing_set_predictions)
precision = precision_score(y_test, testing_set_predictions)
f1 = f1_score(y_test, testing_set_predictions)
print("Performance over the testing data set \n")
print("Accuracy : {} \nRecall : {} \nPrecision : {} \nF1 : {}\n".format(accuracy, recall, precision, f1))
