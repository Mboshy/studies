import scipy.io.wavfile as wfile
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math

org = wfile.read('org.wav')
long = wfile.read('long2.wav')
short = wfile.read('short2.wav')
gowno = np.asarray(org[1])



def plotting():
    fig, a = plt.subplots(3, 1)

    a[0].plot(org[1])
    a[1].plot(long[1])
    a[2].plot(short[1])

    plt.show()


def nn():
    normalized_param = 35768.

    x = np.arange(0, len(org[1]), 1)
    y = np.asarray(org[1][50000:60000])
    short_copy = org[1].copy()
    long_copy = org[1].copy()
    y = np.divide(y, normalized_param)
    new_y = y[100:]
    new_x = [y[i:i+100] for i in range(len(y)-100)]
    new_x = np.array(new_x)
    new_y = np.array(new_y)



    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=100))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(new_x, new_y, epochs=40, validation_split=0.3, batch_size=64)

    rms1 = 0

    for i in range(55536, 55720):
        train_wav = np.array(short_copy[i-100:i])
        train_wav = np.divide(train_wav, normalized_param)
        train_wav = np.expand_dims(train_wav, axis=0)
        predicted_wav = model.predict(train_wav)
        short_copy[i] = predicted_wav * normalized_param
        # rms1 +=

    rms1 = np.sqrt((short_copy - gowno)**2 / 2.0)

    fig, a = plt.subplots(2, 1)
    a[0].plot(org[1][55000:57000])
    a[1].plot(short_copy[55000:57000])
    plt.show()


    for i in range(55536, 56106):
        train_wav = np.array(long_copy[i-100:i])
        train_wav = np.divide(train_wav, normalized_param)
        train_wav = np.expand_dims(train_wav, axis=0)
        predicted_wav = model.predict(train_wav)
        long_copy[i] = predicted_wav * normalized_param

    rms2 = np.sqrt((long_copy - gowno)**2 / 2.0)



    fig, a = plt.subplots(2, 1)
    a[0].plot(org[1][55000:57000])
    a[1].plot(long_copy[55000:57000])
    plt.show()

    print(np.nansum(rms1))
    print(np.nansum(rms2))


def main():
    # plotting()
    nn()


if __name__ == '__main__':
    main()
