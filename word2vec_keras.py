# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:14:08 2020

@author: tamohant
"""
import os
from collections import Counter
from time import time

import numpy as np
import pandas as pd
from keras.layers import Dense, Dot, Embedding, Input, Reshape
from keras.models import Model
from keras.preprocessing.sequence import skipgrams
from nltk.corpus import stopwords

np.random.seed(777)
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

def preprocessing_corpus(corpus, sampling_rate=1.0):
    if sampling_rate is not 1.0:
        corpus = corpus.sample(frac=sampling_rate, replace=False)
    corpus = corpus.str.lower()
    corpus = corpus.str.replace(r'[^A-Za-z0-9\s]', ' ', regex=True)
    return corpus.values.tolist()

def making_vocab(corpus, top_n_ratio=1.0):
    words = np.concatenate(np.core.defchararray.split(corpus)).tolist()

    stopWords = set(stopwords.words('english'))
    words = [word for word in words if word not in stopWords]

    counter = Counter(words)
    if top_n_ratio is not 1.0:
        counter = Counter(dict(counter.most_common(int(top_n_ratio*len(counter)))))
    unique_words = list(counter) + ['UNK']
    return unique_words
    
def vocab_indexing(vocab):
    word2index = {word:index for index, word in enumerate(vocab)}
    index2word = {index:word for word, index in word2index.items()}
    return word2index, index2word

def word_index_into_corpus(word2index, corpus):
    indexed_corpus = []
    for doc in corpus:
        indexed_corpus.append([word2index[word] if word in word2index else word2index['UNK'] for word in doc.split()])
    return indexed_corpus

def generating_wordpairs(indexed_corpus, vocab_size, window_size=4):
    X = []
    Y = []
    for row in indexed_corpus:
        x, y = skipgrams(sequence=row, vocabulary_size=vocab_size, window_size=window_size,
                        negative_samples=1.0, shuffle=True, categorical=False, sampling_table=None, seed=None)
        X = X + list(x)
        Y = Y + list(y)
    return X, Y

def consructing_model(vocab_size, embedding_dim=300):
    input_target = Input((1,))
    input_context = Input((1,))

    embedding_layer = Embedding(vocab_size, embedding_dim, input_length=1)

    target_embedding = embedding_layer(input_target)
    target_embedding = Reshape((embedding_dim, 1))(target_embedding)
    context_embedding = embedding_layer(input_context)
    context_embedding = Reshape((embedding_dim, 1))(context_embedding)

    hidden_layer = Dot(axes=1)([target_embedding, context_embedding])
    hidden_layer = Reshape((1,))(hidden_layer)

    output = Dense(16, activation='sigmoid')(hidden_layer)
    output = Dense(1, activation='sigmoid')(output)
    
    model = Model(inputs=[input_target, input_context], outputs=output)
    #model.summary()
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    return model

def training_model(model, epochs, batch_size, indexed_corpus, vocab_size):
    for i in range(epochs):
        idx_batch = np.random.choice(len(indexed_corpus), batch_size)
        X, Y = generating_wordpairs(np.array(indexed_corpus)[idx_batch].tolist(), vocab_size)

        word_target, word_context = zip(*X)
        word_target = np.array(word_target, dtype=np.int32)
        word_context = np.array(word_context, dtype=np.int32)

        target = np.zeros((1,))
        context = np.zeros((1,))
        label = np.zeros((1,))
        idx = np.random.randint(0, len(Y)-1)
        target[0,] = word_target[idx]
        context[0,] = word_context[idx]
        label[0,] = Y[idx]
        loss = model.train_on_batch([target, context], label)
        if i % 1000 == 0:
            print("Iteration {}, loss={}".format(i, loss))
    return model

def save_vectors(file_path, vocab_size, embedding_dim, model, word2index):
    f = open(file_path, 'w')
    f.write('{} {}\n'.format(vocab_size-1, embedding_dim))
    vectors = model.get_weights()[0]
    for word, i in word2index.items():
        f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
    f.close()
    return file_path

if __name__ == "__main__":
    time_start = time()
    time_check = time()
    
    corpus = pd.read_csv("abcnews-date-text.csv").iloc[:,1] 
    corpus = preprocessing_corpus(corpus, sampling_rate=1.0)
    print("Corpus was loaded in\t{time} sec".format(time=time()-time_check)); time_check = time()
    
    vocab = making_vocab(corpus, top_n_ratio=0.8)
    vocab_size = len(vocab)
    print("Vocabulary was made in\t{time} sec".format(time=time()-time_check)); time_check = time()
    
    word2index, index2word = vocab_indexing(vocab)
    print("Vocabulary was indexed in\t{time} sec".format(time=time()-time_check)); time_check = time()
    
    indexed_corpus = word_index_into_corpus(word2index, corpus)
    print("Corpus was indexed in\t{time} sec".format(time=time()-time_check)); time_check = time()

    embedding_dim = 100
    model = consructing_model(vocab_size, embedding_dim=embedding_dim)
    print("Model was constructed in\t{time} sec".format(time=time()-time_check)); time_check = time()

    epochs = 100001
    batch_sentence_size = 512
    model = training_model(model, epochs, 512, indexed_corpus, vocab_size)
    print("Traning was done in\t{time} sec".format(time=time()-time_check)); time_check = time()

    save_path = save_vectors('vectors_on_batch.txt', vocab_size, embedding_dim, model, word2index)
    print("Trained vector was saved in\t{time} sec".format(time=time()-time_check)); time_check = time()

    print("Done: overall process consumes\t{time} sec".format(time=time()-time_start))
