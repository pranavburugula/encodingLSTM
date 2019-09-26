from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense
import numpy as np
import pandas
from sklearn.datasets import fetch_20newsgroups_vectorized
import os

if __name__ == "__main__":
    categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 
    'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 
    'rec.sport.hockey', 'sci.crypt', 'sci.electronics']
    train = fetch_20newsgroups_vectorized(subset='train', categories=categories)
    test = fetch_20newsgroups_vectorized(subset='test', categories=categories)
    model = Sequential()
    createModel(model, [200, 400, 20], train.data.shape)
    trainModel(model, train.data, train.target)
    saveModel(model, 'C:\\Users\\pb8xe\\Documents\\C3SR\\encodingLSTM\\models', '20newsgroupsLSTM')
    evaluateModel(model, test.data, test.target)
    pass

def createModel(model, model_structure, input_shape):
    model.add(Bidirectional(LSTM(model_structure[0], return_sequences=True), input_shape=input_shape, merge_mode='concat'))
    for i in range(1, len(model_structure) - 1):
        model.add(LSTM(model_structure[i], return_sequences=True))
    model.add(Dense(model_structure[len(model_structure) - 1]))
    model.compile(loss='mean_absolute_percentage_error', optimizer='sgd', metrics=['accuracy', 'recall', 'precision'])

def trainModel(model, X, y, epochs, batch_size, validation_split):
    model.fit(X, y, epochs=epochs, validation_split=validation_split, batch_size=batch_size)

def evaluateModel(model, X, y):
    score = model.evaluate(X, y, verbose=2)
    print('Accuracy: ', score[0])
    print('Recall: ', score[1])
    print('Precision: ', score[2])

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