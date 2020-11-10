import numpy as np
from nltk.corpus import brown, treebank
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras import Sequential
from keras.layers import Dense, Flatten, LSTM


def evaluate_tagger():
    words, tags = get_corpus()
    predicted_tags = pos_tag(words)
    print('* POS tagger accuracy: {:.2%}'.format(accuracy_score(tags,
                                                            predicted_tags)))


def pos_tag(corpus):
    pos_tagger = train_tagger()
    tagged_corpus = pos_tagger.predict(corpus)
    return tagged_corpus


def train_tagger():
    x, y = load_dataset(brown)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
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


def load_dataset(corpus):
    dataset = set(corpus.tagged_words(tagset='universal'))
    x = [word for (word, tag) in dataset]
    y = [tag for (word, tag) in dataset]
    return x, y


def get_corpus():
    return load_dataset(treebank)


def main():
    words, true_tags = get_corpus()
    predicted_tags = pos_tag(words)
    evaluate_tagger()


if __name__ == '__main__':
    main()