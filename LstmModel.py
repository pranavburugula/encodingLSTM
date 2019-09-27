from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense
import numpy as np
import pandas
from sklearn.datasets import fetch_20newsgroups_vectorized
import os

def createModel(model, model_structure, input_shape):
    # model.add(Bidirectional(LSTM(model_structure[0], return_sequences=True), input_shape=input_shape, merge_mode='concat'))
    model.add(LSTM(model_structure[0], return_sequences=True, input_shape=input_shape))
    for i in range(1, len(model_structure) - 1):
        model.add(LSTM(model_structure[i], return_sequences=True))
    model.add(Dense(model_structure[len(model_structure) - 1]))
    model.compile(loss='mean_absolute_percentage_error', optimizer='sgd', metrics=['accuracy'])

def trainModel(model, X, y, epochs, batch_size, validation_split):
    model.fit(X, y, epochs=epochs, validation_split=validation_split, batch_size=batch_size)

def evaluateModel(model, X, y):
    score = model.evaluate(X, y, verbose=2)
    print('Accuracy: ', score[0])
    # print('Recall: ', score[1])
    # print('Precision: ', score[2])

def saveModel(model, outputDir, filePrefix):
        outFilename_model = filePrefix + '.json'
        outFilepath = os.path.join(outputDir, outFilename_model)
        # serialize model to JSON
        model_json = model.to_json()
        with open(outFilepath, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        outFilename_weights = filePrefix + '.h5'
        outFilepath = os.path.join(outputDir, outFilename_weights)
        model.save_weights(outFilepath)
        print("Saved model to disk")

if __name__ == "__main__":
    categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 
    'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 
    'rec.sport.hockey', 'sci.crypt', 'sci.electronics']
    train_source = fetch_20newsgroups_vectorized(subset='train')
    X_train = np.array(train_source.data)
    y_train = np.array(train_source.target)
    print('X train: ',X_train)
    # print('y train: ', y_train)
    print('X train shape: ', X_train.shape)
    X_train = X_train.reshape((1, X_train.shape[0], X_train.shape[1]))
    print('X train shape after reshaping: ', X_train.shape)
    print('y train shape: ', y_train.shape)

    test_source = fetch_20newsgroups_vectorized(subset='test')
    X_test = np.array(test_source.data)
    y_test = np.array(test_source.target)
    # print('X test: ',X_test)
    # print('y test: ', y_test)
    print('X test shape: ', X_test.shape)
    X_test = X_test.reshape((1, X_test.shape[0], X_test.shape[1]))
    print('X test shape after reshaping: ', X_test.shape)
    print('y test shape: ', y_test.shape)
    model = Sequential()

    # X_train_reshaped = X_train.reshape((1, X_train.size, X_train[0].size))
    # y_train_reshaped = y_train.reshape((1, y_train.size, y_train[0].size))

    # X_test_reshaped = X_test.reshape((1, X_test.size, X_test[0].size))
    # y_test_reshaped = y_test.reshape((1, y_test.size, y_test[0].size))

    createModel(model, [200, 20], input_shape=(X_train.shape[1], X_train.shape[2]))
    trainModel(model, X_train, y_train, epochs=100, batch_size=1, validation_split=0.33)
    saveModel(model, 'C:\\Users\\pb8xe\\Documents\\C3SR\\encodingLSTM\\models', '20newsgroupsLSTM')
    evaluateModel(model, X_test, y_test)
    pass