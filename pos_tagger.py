import numpy as np
import nltk
import sklearn
import keras


def pos_tag(corpus):
    pos_tagger = train_tagger()


def train_tagger():
    x_train, x_test, y_train, y_test = load_dataset()


def load_dataset():
    corpus = set(nltk.corpus.brown.tagged_words(tagset='universal'))
    x = [word for (word, tag) in corpus]
    y = [tag for (word, tag) in corpus]
    x_train, x_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(x, y, test_size=0.25)
    return x_train, x_test, y_train, y_test


def get_corpus():
    return


def main():
    corpus = get_corpus()
    pos_tag(corpus)


if __name__ == '__main__':
    main()