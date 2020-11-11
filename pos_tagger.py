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
    words, tags = load_dataset(treebank)
    words = sequence.pad_sequences(words, maxlen=56057)

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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    model = compile_network()
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              batch_size=128, epochs=1)
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
    words = [word for (word, tag) in dataset]
    tags = [tag for (word, tag) in dataset]
    x = encode_array(words)
    y = encode_array(tags)
    return x, y


def encode_array(array):
    reshaped_array = np.array(array).reshape(-1, 1)
    encoded_array = OneHotEncoder().fit_transform(reshaped_array).toarray()
    return encoded_array


def main():
    evaluate_tagger()


if __name__ == '__main__':
    main()