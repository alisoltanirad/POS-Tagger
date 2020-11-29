# https://github.com/alisoltanirad/POS-Tagger
import numpy as np
from nltk.corpus import brown, treebank
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, Dense, Flatten, LSTM


class POS_Tagger():

    def __init__(self):
        self._tagger = self._train_tagger()

    def tag(self, corpus):
        words = self._preprocess_input(corpus)
        return self._tagger.predict(words)

    def evaluate(self):
        words, tags = self._load_dataset(treebank)

        predicted_tags = self.tag(words)
        predicted_tags = (predicted_tags > 0.5)

        evaluation_data = {
            'Accuracy': accuracy_score(tags, predicted_tags),
        }
        return evaluation_data

    def _train_tagger(self):
        x, y = self._load_dataset(brown)
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.25)
        model = self._compile_network()
        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  batch_size=256, epochs=2, verbose=2)
        return model

    def _compile_network(self):
        network = self._create_network()
        network.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
        return network

    def _create_network(self):
        network = Sequential()
        network.add(Dense(8))
        network.add(Dense(16))
        network.add(Dense(12, activation='softmax'))
        return network

    def _preprocess_input(self, corpus):
        return sequence.pad_sequences(corpus, maxlen=len(set(brown.words())))

    def _load_dataset(self, corpus):
        dataset = set(corpus.tagged_words(tagset='universal'))
        words = [word for (word, tag) in dataset]
        tags = [tag for (word, tag) in dataset]
        x = self._encode_array(words)
        y = self._encode_array(tags)
        return x, y

    def _encode_array(self, array):
        reshaped_array = np.array(array).reshape(-1, 1)
        encoded_array = OneHotEncoder().fit_transform(reshaped_array).toarray()
        return encoded_array


def main():
    tagger = POS_Tagger()
    evaluation_data = tagger.evaluate()
    print('* POS Tagger Accuracy: {:.2%}'.format(evaluation_data['Accuracy']))


if __name__ == '__main__':
    main()