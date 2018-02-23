import gensim
import tensorflow as tf
import pandas as pd
import nltk
import string
import itertools
import re
from sklearn.model_selection import train_test_split
from keras.engine import Layer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import word2vec
from lstm_test import create_features
from word2vec import clean_data

from keras.models import Model
from keras.layers import Dense, Embedding, Input, GRU, Flatten, concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D, \
    SpatialDropout1D
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

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

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))


def get_model(vector_size, word_count, max_sequence_length, embedding_weights, second_input_len):
    inp = Input(shape=(max_sequence_length,))
    embedding_layer = Embedding(word_count+1,
                                vector_size,
                                weights=[embedding_weights],
                                input_length=max_sequence_length,
                                trainable=False)(inp)
    # x = SpatialDropout1D(0.2)(embedding_layer)
    x = Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=0.2))(embedding_layer)
    # x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    additional_input = Input(shape=(second_input_len,), name='second_input')
    x = concatenate([conc, additional_input])
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=[inp, additional_input], outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


max_len = 300


def train():
    second_input = create_features('train.csv')
    print(second_input.shape)
    train = pd.read_csv("train.csv")
    train = train.sample(frac=1)

    list_sentences_train = train["comment_text"].fillna("CVxTz").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values

    word2vec = gensim.models.Word2Vec.load('w2v.model')

    forward_index, invert_index = make_index(word2vec)


    train_sentences = list()

    for sentence in tqdm(list_sentences_train, desc='normalize sentences'):
        norm_sentence = convert_sentence_to_indexes(sentence, invert_index, word2vec, max_len)

        train_sentences.append(norm_sentence)

    train_sentences = np.array(train_sentences)

    embedding_matrix = np.zeros((len(word2vec.wv.vocab)+1, word2vec.vector_size))
    for word, i in tqdm(invert_index.items(), desc='create embedding weights'):
        if word in word2vec.wv.vocab:
            embedding_matrix[i] = word2vec.wv[word]

    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    model = get_model(word2vec.vector_size, len(word2vec.wv.vocab), max_len, embedding_matrix, second_input.shape[1])

    batch_size = 64
    epochs = 10

    file_path = "weights_base.best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)


    exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1
    steps = int(len(train_sentences) / batch_size) * epochs
    lr_init, lr_fin = 0.001, 0.0005
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    K.set_value(model.optimizer.lr, lr_init)
    K.set_value(model.optimizer.decay, lr_decay)

    #[X_tra, X_val, y_tra, y_val] = train_test_split(train_sentences, y, train_size=0.9, random_state=233)
    #[X_tra, X_val, y_tra, y_val] = train_test_split(train_sentences, y, train_size=0.9, random_state=233)
    #RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
    callbacks_list = [checkpoint, early]  # early

    model.fit([train_sentences, second_input], y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)


def predict():
    test = pd.read_csv("test.csv")

    list_sentences_test = test["comment_text"].fillna("CVxTz").values

    word2vec = gensim.models.Word2Vec.load('w2v.model')

    forward_index, invert_index = make_index(word2vec)
    train_sentences = list()

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
