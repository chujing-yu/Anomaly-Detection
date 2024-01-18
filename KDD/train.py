import numpy as np

from methods import *

x_train = pd.read_csv('dataset/x_train.csv')
y_train = pd.read_csv('dataset/y_train.csv')
x_test = pd.read_csv('dataset/x_test.csv')
y_test = pd.read_csv('dataset/y_test.csv')



from tensorflow.python.keras.optimizer_v2 import adam


def getModel():
    input_layer = Input(shape=(x_train.shape[1],))
    encoded = Dense(8, activation='relu', activity_regularizer=kr.regularizers.l2(10e-5))(input_layer)  # l2正则化约束
    decoded = Dense(x_train.shape[1], activation='relu')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=adam.Adam(), loss='mean_squared_error')
    return autoencoder


norm = np.where(y_train['label'] == 0)[0]

autoencoder = getModel()
history = autoencoder.fit(
    x_train.iloc[norm], x_train.iloc[norm],
    epochs=10,
    batch_size=100,
    shuffle=False,
    validation_split=0.1
)
autoencoder.save('results/autoencoder_epoch_10_batch_100_valid_0.1_.h5')
history_df = pd.DataFrame(history.history)
history_df.to_csv('results/training_history_epoch_10_batch_100_valid_0.1_.csv', index=False)
# threshold = history.history["loss"][-1]
# print(threshold)
