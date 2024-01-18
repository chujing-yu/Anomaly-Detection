from methods import *

x_test = pd.read_csv('dataset/x_test.csv')
y_test = pd.read_csv('dataset/y_test.csv')
autoencoder = load_model('results/autoencoder_epoch_10_batch_100_valid_0.1_.h5')
history = pd.read_csv('results/training_history_epoch_10_batch_100_valid_0.1_.csv')

# 我们将阈值设置为等于自动编码器的训练损失
threshold = history["loss"].tolist()[-1]
testing_set_predictions = autoencoder.predict(x_test)
test_losses = calculate_losses(x_test, testing_set_predictions)

testing_set_predictions = np.zeros(len(test_losses))
testing_set_predictions[np.where(test_losses > threshold)] = 1

recall = recall_score(y_test, testing_set_predictions)
precision = precision_score(y_test, testing_set_predictions)
f1 = f1_score(y_test, testing_set_predictions)
print("Performance over the testing data set \n")
print("Accuracy : {} \nRecall : {} \nPrecision : {} \nF1 : {}\n".format(accuracy, recall, precision, f1))
