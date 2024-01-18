from model import *

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

x_test = pd.read_csv('processed/' + dataset + '/x_test.csv')
y_test = pd.read_csv('processed/' + dataset + '/y_test.csv')

input_dim = x_train.shape[1]

vae = TimeSeriesVAE(input_dim, latent_dim, intermediate_dim).vae_model

vae.load_weights('results/' + dataset + '/' + model_filename + '.h5')

Q = np.arange(80, 101)
reconstructed_data = vae.predict(x_test)
mse = np.mean(np.power(x_test - reconstructed_data, 2), axis=1)
test_losses = mse
testing_set_predictions = np.zeros(len(test_losses))

for q in Q:
    print(q)
    threshold = np.percentile(mse, q)  # 设置异常检测的阈值

    testing_set_predictions[np.where(test_losses > threshold)] = 1

    recall = recall_score(y_test, testing_set_predictions)
    precision = precision_score(y_test, testing_set_predictions)
    f1 = f1_score(y_test, testing_set_predictions)
    print("Performance over the testing data set \n")
    print("Accuracy : {} \nRecall : {} \nPrecision : {} \nF1 : {}\n".format(accuracy, recall, precision, f1))
