from methods import *


def Autoencoder(x_train):
    input_layer = Input(shape=(x_train.shape[1],))
    encoded = Dense(1028, activation='relu', kernel_initializer='he_uniform',
                    activity_regularizer=kr.regularizers.l2(10e-4))(input_layer)  # l2正则化约束
    decoded = Dense(x_train.shape[1], activation='relu')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder


# original_dim = x_train.shape[1]
# latent_dim = 2
# intermediate_dim = 256
#
# encoder_inputs = Input(shape=(original_dim,))
# h = layers.Dense(intermediate_dim, activation='relu')(encoder_inputs)
# # 计算p(Z|X)的均值和方差
# z_mean = layers.Dense(latent_dim)(h)
# z_log_var = layers.Dense(latent_dim)(h)


# # 重参数技巧
# def sampling(args):
#     z_mean, z_log_var = args
#     epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
#     return z_mean + tf.exp(0.5 * z_log_var) * epsilon
#
#
# # 重参数层，相当于给输入加入噪声
# z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
#
# # 解码层
# """decoder_outputs = layers.Dense(train_data.shape[1], activation='sigmoid')(x)
#
# decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
#
# vae = keras.Model(encoder_inputs, decoder(encoder(encoder_inputs)[2]), name='vae')"""
# decoder_h = layers.Dense(intermediate_dim, activation='relu', kernel_regularizer=kr.regularizers.l2(0.01))
# decoder_mean = layers.Dense(original_dim, activation='sigmoid')
# h_decoded = decoder_h(z)
# x_decoded_mean = decoder_mean(h_decoded)
#
# # 建立模型
# vae = kr.Model(encoder_inputs, x_decoded_mean)
#

#
# # 编译并训练VAE模型
# vae.compile(optimizer='adam')
# vae.fit(x_train,
#         shuffle=True,
#         epochs=50,
#         batch_size=100,
#         validation_data=(x_test, None))


class TimeSeriesVAE:
    def __init__(self, input_dim, latent_dim, intermediate_dim):
        self.original_dim = input_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.vae_model = self.build_vae_model()

    def build_vae_model(self):
        encoder_inputs = Input(shape=(self.original_dim,))
        h = layers.Dense(self.intermediate_dim, activation='relu')(encoder_inputs)
        # 计算p(Z|X)的均值和方差
        z_mean = layers.Dense(self.latent_dim)(h)
        z_log_var = layers.Dense(self.latent_dim)(h)

        # 重参数技巧
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.latent_dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        # 重参数层，相当于给输入加入噪声
        z = layers.Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

        decoder_input = layers.Dense(self.intermediate_dim, activation='relu'
                                     # ,kernel_regularizer=kr.regularizers.l2(10e-4)
                                     )(z)
        decoder_outputs = layers.Dense(self.original_dim, activation='sigmoid')(decoder_input)


        # 定义VAE的损失函数
        reconstruction_loss = backend.sum(backend.binary_crossentropy(encoder_inputs, decoder_outputs), axis=-1)
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss) * -0.5

        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        # 建立模型
        vae = Model(encoder_inputs, decoder_outputs)
        vae.add_loss(vae_loss)
        # 编译并训练VAE模型
        vae.compile(optimizer='adam')

        return vae
