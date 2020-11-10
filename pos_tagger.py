import numpy as np
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, Flatten, LSTM


def pos_tag(corpus):
    pos_tagger = train_tagger()


def train_tagger():
    x_train, x_test, y_train, y_test = load_dataset()



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
    pos_tag(corpus)


if __name__ == '__main__':
    main()