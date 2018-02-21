import gensim
import tensorflow as tf
import pandas as pd
import nltk
import string
import itertools
import re

from keras.engine import Layer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
import numpy as np
from tqdm import tqdm

from word2vec import clean_data

from keras.models import Model
from keras.layers import Dense, Embedding, Input, GRU
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import initializers
from keras import backend as K

stop_words = set(stopwords.words('english'))


def preprocessing_text(lst_with_text):
    lst_new = []
    available_symbols = 'qwertyuiopasdfghjklzxcvbnm.?!"\' -'
    replaced = set()
    for el in lst_with_text:
        tmp_str = el.lower()
        tmp_str2 = tmp_str
        for sym in tmp_str:
            if sym not in available_symbols:
                tmp_str2 = tmp_str2.replace(sym, ' ')
                replaced.add(sym)
        lst_new.append(tmp_str2)
    return lst_new


def aaaa():
    # This Python 3 environment comes with many helpful analytics libraries installed
    # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
    # For example, here's several helpful packages to load in

    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    # Input data files are available in the "../input/" directory.
    # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

    from subprocess import check_output


    # Any results you write to the current directory are saved as output.

    from keras.models import Model
    from keras.layers import Dense, Embedding, Input
    from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
    from keras.preprocessing import text, sequence
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    max_features = 20000
    maxlen = 300


    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    train = train.sample(frac=1)

    list_sentences_train = train["comment_text"].fillna("CVxTz").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values
    list_sentences_test = test["comment_text"].fillna("CVxTz").values


    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

    test = text.text_to_word_sequence(list_sentences_train)
    X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

    def get_model():
        embed_size = 256
        inp = Input(shape=(300, ))
        x = Bidirectional(LSTM(50, return_sequences=True))(inp)
        x = GlobalMaxPool1D()(x)
        x = Dropout(0.4)(x)
        x = Dense(50, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model


    model = get_model()
    batch_size = 32
    epochs = 20


    file_path="weights_base.best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)


    callbacks_list = [checkpoint, early] #early
    #model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

    model.load_weights(file_path)

    y_test = model.predict(X_te)



    sample_submission = pd.read_csv("sample_submission.csv")

    sample_submission[list_classes] = y_test

    sample_submission.to_csv("baseline.csv", index=False)
    #lst_true = preprocessing_text(lst_true)
    #lst_false = preprocessing_text(lst_false)


def convert_sentence_to_indexes(text, inverse_index_dict, word2vec_model, max_len):
    filtered = [x for x in clean_data(text).split() if x]

    res = [inverse_index_dict[word] for word in filtered if word in word2vec_model.wv.vocab]
    if len(res) > max_len:
        res = res[:max_len]
    elif len(res) < max_len:
        res = np.concatenate((res, np.array([len(word2vec_model.wv.vocab)] * (max_len - len(res)))))
    return res


def make_index(word2vec_model):
    index_dict = dict()
    inverse_index_dict = dict()
    index = 0
    for word in tqdm(word2vec_model.wv.vocab, desc='create index'):
        index_dict[index] = word
        inverse_index_dict[word] = index
        index += 1
    return index_dict, inverse_index_dict


def get_model(vector_size, word_count, max_sequence_length, embedding_weights):
    inp = Input(shape=(max_sequence_length,))
    embedding_layer = Embedding(word_count+1,
                                vector_size,
                                weights=[embedding_weights],
                                input_length=max_sequence_length,
                                trainable=False)(inp)
    x = Bidirectional(GRU(100, return_sequences=True))(embedding_layer)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.4)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        super(AttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.init((input_shape[-1],))
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')

        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

def train():
    train = pd.read_csv("train.csv")

    train = train.sample(frac=1)

    list_sentences_train = train["comment_text"].fillna("CVxTz").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values

    word2vec = gensim.models.Word2Vec.load('w2v.model')

    forward_index, invert_index = make_index(word2vec)


    train_sentences = list()
    max_len = 300
    for sentence in tqdm(list_sentences_train, desc='normalize sentences'):
        norm_sentence = convert_sentence_to_indexes(sentence, invert_index, word2vec, max_len)

        train_sentences.append(norm_sentence)


    train_sentences = np.array(train_sentences)


    embedding_matrix = np.zeros((len(word2vec.wv.vocab)+1, word2vec.vector_size))
    for word, i in tqdm(invert_index.items(), desc='create embedding weights'):
        if word in word2vec.wv.vocab:
            embedding_matrix[i] = word2vec.wv[word]

    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


    model = get_model(word2vec.vector_size, len(word2vec.wv.vocab), max_len, embedding_matrix)

    batch_size = 32
    epochs = 20

    file_path = "weights_base.best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

    callbacks_list = [checkpoint, early]  # early
    model.fit(train_sentences, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

def predict():
    test = pd.read_csv("test.csv")

    list_sentences_test = test["comment_text"].fillna("CVxTz").values

    word2vec = gensim.models.Word2Vec.load('w2v.model')

    forward_index, invert_index = make_index(word2vec)
    train_sentences = list()
    max_len = 300
    for sentence in tqdm(list_sentences_test, desc='normalize sentences'):
        norm_sentence = convert_sentence_to_indexes(sentence, invert_index, word2vec, max_len)

        train_sentences.append(norm_sentence)

    train_sentences = np.array(train_sentences)

    embedding_matrix = np.zeros((len(word2vec.wv.vocab) + 1, word2vec.vector_size))
    for word, i in tqdm(invert_index.items(), desc='create embedding weights'):
        if word in word2vec.wv.vocab:
            embedding_matrix[i] = word2vec.wv[word]


    model = get_model(word2vec.vector_size, len(word2vec.wv.vocab), max_len, embedding_matrix)
    file_path = "weights_base.best.hdf5"

    model.load_weights(file_path)

    y_test = model.predict(train_sentences)

    sample_submission = pd.read_csv("sample_submission.csv")
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    sample_submission[list_classes] = y_test

    sample_submission.to_csv("baseline.csv", index=False)

train()