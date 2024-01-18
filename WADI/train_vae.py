from methods import *
from model import *
import tensorflow as tf
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras import layers, backend

with open('config_vae.json') as config_file:
    config = json.load(config_file)

dataset = config['dataset']
model_name = config['model_name']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
valid_rate = config['valid_rate']
latent_dim = config['latent_dim']
intermediate_dim = config['intermediate_dim']

model_filename = f"{model_name}_batch_{batch_size}_epoch_{num_epochs}_valid_{valid_rate}"

if dataset == 'wadi':
    x_train = pd.read_csv('processed/wadi/x_train.csv')
    x_train = x_train.iloc[1:, :]
    print(x_train.shape)
else:
    x_train = pd.read_csv('processed/kdd/x_train.csv')
    y_train = pd.read_csv('processed/kdd/y_train.csv')
    norm = np.where(y_train['label'] == 0)[0]
    x_train = x_train.iloc[norm]
    print(x_train.shape)

# x_test = pd.read_csv('processed/' + dataset + '/x_test.csv')
# y_test = pd.read_csv('processed/' + dataset + '/y_test.csv')
print('start training....')

# 创建 TimeSeriesVAE 实例并使用
input_dim = x_train.shape[1]

vae = TimeSeriesVAE(input_dim, latent_dim, intermediate_dim).vae_model

# 训练模型
vae.fit(x_train,
        shuffle=True,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=valid_rate)

vae.save_weights('results/' + dataset + '/' + model_filename + '.h5')

print('training completed')
