from keras.models import LSTM, Sequential, Bidirectional, Dense
import numpy as np
import pandas
from sklearn.datasets import fetch_20newsgroups_vectorized

if __name__ == "__main__":
    train = fetch_20newsgroups_vectorized(subset='train')
    test = fetch_20newsgroups_vectorized(subset='test')
    model = Sequential()
    createModel(model, [])
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
