import numpy as np
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, Flatten, LSTM


def pos_tag(corpus):
    pos_tagger = train_tagger()
    tagged_corpus = pos_tagger.predict(corpus)
    return tagged_corpus


def train_tagger():
    x_train, x_test, y_train, y_test = load_dataset()
    model = compile_network()
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              batch_size=8, epochs=1)
    return model


def compile_network():
    network = create_network()
    network.compile(loss='categorical_crossentropy', optimizer='adam')
    return network


def create_network():
    network = Sequential()
    network.add(LSTM(128))
    network.add(LSTM(64))
    network.add(LSTM(64))
    network.add(LSTM(12, activation='softmax', recurrent_activation='softmax'))
    return network


def load_dataset():
    corpus = set(brown.tagged_words(tagset='universal'))
    x = [word for (word, tag) in corpus]
    y = [tag for (word, tag) in corpus]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    return x_train, x_test, y_train, y_test


def get_corpus():
    return


def main():
    corpus = get_corpus()
    tagged_corpus = pos_tag(corpus)


if __name__ == '__main__':
    main()