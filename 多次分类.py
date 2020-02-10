from keras.datasets import imdb
from keras import backend as K
from keras.layers import LSTM,Conv1D,Dense
from keras.models import Sequential
from keras.layers import Bidirectional
import numpy as np
from keras.layers import Input,Embedding,Lambda
from keras.models import Model
import multiprocessing
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
import gensim
import os
from keras.layers import *

from keras.models import Model

import keras.backend as K

from keras.callbacks import Callback



def sent2vec(S):

    """S格式：[[w1, w2]]

    """

    V = []

    for s in S:

        V.append([])

        for w in s:

            for _ in w:

                V[-1].append(word2id.get(w, 0))

    V = seq_padding(V)

    V = word2vec[V]

    return V

class OurLayer(Layer):

    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层

    """

    def reuse(self, layer, *args, **kwargs):

        if not layer.built:

            if len(args) > 0:

                inputs = args[0]

            else:

                inputs = kwargs['inputs']

            if isinstance(inputs, list):

                input_shape = [K.int_shape(x) for x in inputs]

            else:

                input_shape = K.int_shape(inputs)

            layer.build(input_shape)

        outputs = layer.call(*args, **kwargs)

        for w in layer.trainable_weights:

            if w not in self._trainable_weights:

                self._trainable_weights.append(w)

        for w in layer.non_trainable_weights:

            if w not in self._non_trainable_weights:

                self._non_trainable_weights.append(w)

        return outputs




class DilatedGatedConv1D(OurLayer):

    """膨胀门卷积（DGCNN）

    """

    def __init__(self,

                 o_dim=None,

                 k_size=3,

                 rate=1,

                 skip_connect=True,

                 drop_gate=None,

                 **kwargs):

        super(DilatedGatedConv1D, self).__init__(**kwargs)

        self.o_dim = o_dim

        self.k_size = k_size

        self.rate = rate

        self.skip_connect = skip_connect

        self.drop_gate = drop_gate

    def build(self, input_shape):
        super(DilatedGatedConv1D, self).build(input_shape)
        if self.o_dim is None:
            self.o_dim = input_shape[0][-1]
        self.conv1d = Conv1D(
            self.o_dim * 2,
            self.k_size,
            dilation_rate=self.rate,
            padding='same'
        )
        if self.skip_connect and self.o_dim != input_shape[0][-1]:
            self.conv1d_1x1 = Conv1D(self.o_dim, 1)
    def call(self, inputs):
        xo, mask = inputs
        x = xo * mask
        x = self.reuse(self.conv1d, x)
        x, g = x[..., :self.o_dim], x[..., self.o_dim:]
        if self.drop_gate is not None:
            g = K.in_train_phase(K.dropout(g, self.drop_gate), g)
        g = K.sigmoid(g)
        if self.skip_connect:
            if self.o_dim != K.int_shape(xo)[-1]:
                xo = self.reuse(self.conv1d_1x1, xo)
            return (xo * (1 - g) + x * g) * mask
        else:
            return x * g * mask
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.o_dim,)


def position_id(x):
    if isinstance(x, list) and len(x) == 2:
        x, r = x
    else:
        r = 0
    pid = K.arange(K.shape(x)[1])
    pid = K.expand_dims(pid, 0)
    pid = K.tile(pid, [K.shape(x)[0], 1])
    return K.abs(pid - K.cast(r, 'int32'))

def seq_padding(ML,X, padding=0):#未转化未词向量的pad
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

def vseq_padding(ML,X, padding=0):#转化成词向量的pad
    for i in range(0,len(X)):
        if len(X[i])<ML:
            c=ML-len(X[i])
            me=np.zeros(shape=(c,150),dtype="float32")
            X[i].extend(me)
        else:
            print("got"+str(i),len(X[i]))
    return X
def word2vec(in_txt):
    path = get_tmpfile("word2vec_lstm.model")
    model = Word2Vec(in_txt, size=150, window=6, min_count=1,
                     workers=multiprocessing.cpu_count())
    model.save("word2vec_lstm.model")
def data_set():
    (x_train_in, y_train), (x_test_in, y_test) = imdb.load_data()
    print("x_train",len(x_train_in))
    print("x_test",len(x_test_in))
    x_train_in=x_train_in[:10000]
    x_test_in=x_test_in[:10000]
    y_train=y_train[:10000]
    y_test=y_test[:10000]
    x_train_in_int=x_train_in
    x_test_in_int=x_test_in
    maxlen=0
    averagelen=0
    vocall={}
    x_train=[]
    x_test=[]
    x_train_un=[]
    x_test_un=[]
    if not (os.path.exists("word2vec_lstm.model")):
        print("start train word2vec......")
        for i in range(0,len(x_train_in)):
            for i2 in range(0,len(x_train_in[i])):
                x_train_in[i][i2]=str(x_train_in[i][i2])
        for i in range(0,len(x_test_in)):
            for i2 in range(0,len(x_test_in[i])):
                x_test_in[i][i2]=str(x_test_in[i][i2])
        word2vec(x_train_in+x_test_in)
    model=Word2Vec.load("word2vec_lstm.model")



    for i in range(0,len(x_train_in)):
        senten=[]
        senten_un=[]
        for i2 in range(0,len(x_train_in[i])):
            try:
                vector = model.wv[str(x_train_in[i][i2])]
                senten.append(vector)
                senten_un.append(x_train_in_int[i][i2])
            except:
                continue
        x_train.append(senten)
        x_train_un.append(senten_un)


    for i in range(0,len(x_test_in)):
        senten=[]
        senten_un=[]
        for i2 in range(0,len(x_test_in[i])):
            try:
                vector = model.wv[str(x_test_in[i][i2])]
                senten.append(vector)
                senten_un.append(x_test_int[i][i2])
            except:
                continue
        x_test.append(senten)
        x_test_un.append(senten_un)

    L1 = [len(x) for x in x_train_un]
    ML1 = max(L1)
    L2 = [len(x) for x in x_test_un]
    ML2 = max(L2)
    if ML1>ML2:
        maxlen=ML1
    else:
        maxlen=ML2

    print("maxlen",maxlen)
    
    #加0补齐
    print("type_vdata",type(x_train[1][1][1]))
    x_train=vseq_padding(maxlen,x_train)
    x_test=vseq_padding(maxlen,x_test)
    x_train_un=seq_padding(maxlen,x_train_un)
    x_test_un=seq_padding(maxlen,x_test_un)

    print("shape_x_train",np.shape(x_train[1]))
    print("shape_y_train",np.shape(y_train[1]))
    print("shape_x_train_un",np.shape(x_train_un))
    print("len_words",len(x_train[1]))
    print("len_words",len(x_test[1]))
    word_size=150
    return x_train,x_train_un,y_train,x_test,x_test_un,y_test,maxlen,word_size

def build_model(maxlen,word_size):
    x_in= Input(shape=(None,word_size))
    x_in_un = Input(shape=(None,))
    pid = Lambda(position_id)(x_in_un)
    position_embedding = Embedding(maxlen, word_size, embeddings_initializer='zeros')
    pv = position_embedding(pid)
    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x_in_un)
    x=Dense(word_size,use_bias=False)(x_in)
    x = Add()([x, pv])
    x = DilatedGatedConv1D(rate=1, drop_gate=0.1)([x, mask])
    x = DilatedGatedConv1D(rate=2, drop_gate=0.1)([x, mask])
    x = DilatedGatedConv1D(rate=1, drop_gate=0.1)([x, mask])
    p = Dense(1, activation='sigmoid')(x)
    model=Model([x_in,x_in_un],p)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    return model


if __name__ == "__main__":
    x_train,x_train_un,y_train,x_test,x_test_un,y_test,maxlen,word_size=data_set()
    model=build_model(maxlen,word_size)
    history=model.fit([x_train,x_train_un],y_train,epochs=2,batch_size=32)
    loss,accuracy=model.evaluate([x_test,x_test_un],y_test)
    print(loss,accuracy)