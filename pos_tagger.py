# https://github.com/alisoltanirad/POS-Tagger
import numpy as np
from nltk.corpus import brown, treebank
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, Dense, Flatten, LSTM


def evaluate_tagger():
    words, tags = get_corpus()
    words = np.array(words).reshape(-1, 1)
    words = OneHotEncoder().fit_transform(words).toarray()
    words = sequence.pad_sequences(words, maxlen=56057)
    tags = np.array(tags).reshape(-1, 1)
    tags = OneHotEncoder().fit_transform(tags).toarray()
    predicted_tags = pos_tag(words)
    predicted_tags = (predicted_tags > 0.5)
    print('* POS tagger accuracy: {:.2%}'.format(accuracy_score(tags,
                                                            predicted_tags)))


def pos_tag(corpus):
    pos_tagger = train_tagger()
    tagged_corpus = pos_tagger.predict(corpus)
    return tagged_corpus


def train_tagger():
    x, y = load_dataset(brown)
    x = np.array(x).reshape(-1, 1)
    x = OneHotEncoder().fit_transform(x).toarray()
    y = np.array(y).reshape(-1, 1)
    y = OneHotEncoder().fit_transform(y).toarray()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    model = compile_network()
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              batch_size=1024, epochs=1)
    return model


def compile_network():
    network = create_network()
    network.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return network


def create_network():
    network = Sequential()
    network.add(Dense(8))
    network.add(Dense(16))
    network.add(Dense(12, activation='softmax'))
    return network


def load_dataset(corpus):
    dataset = set(corpus.tagged_words(tagset='universal'))
    x = [word for (word, tag) in dataset]
    y = [tag for (word, tag) in dataset]
    return x, y


def get_corpus():
    return load_dataset(treebank)


def main():
    evaluate_tagger()


if __name__ == '__main__':
    main()