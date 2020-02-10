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

def seq_maxpool(x):

    """seq是[None, seq_len, s_size]的格式，

    mask是[None, seq_len, 1]的格式，先除去mask部分，

    然后再做maxpooling。

    """

    seq, mask = x

    seq -= (1 - mask) * 1e10

    return K.max(seq, 1, keepdims=True)

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

def spilt_4(x,index):
    if index==1:
        return x[:,:250]
    if index==2:
        return x[:,250:500]
    if index==3:
        return x[:,500:750]
    if index==4:
        return x[:,750:]

def union_4(x):
    x1,x2,x3,x4=x
    return K.concatenate([x1,x2,x3,x4],axis=-2)

def mask_ed(x):
    x,mask=x
    return x*mask

def seq_padding(ML,X, padding=0):#未转化未词向量的pad
    for i in range(0,len(X)):
        if len(X[i])<ML:
            c=ML-len(X[i])
            me=np.zeros(shape=(c,1),dtype="float32")
            X[i].extend(me)
        else:
            X[i]=X[i][:ML]
    return X

def vseq_padding(ML,X, padding=0):#转化成词向量的pad
    for i in range(0,len(X)):
        if len(X[i])<ML:
            c=ML-len(X[i])
            me=np.zeros(shape=(c,150),dtype="float32")
            X[i].extend(me)
        else:
            X[i]=X[i][:ML]
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
    maxlen=1000
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

    y_train=np.reshape(y_train,(10000,1,1))
    y_test=np.reshape(y_test,(10000,1,1))
    print("shape_y_train",np.shape(y_train))
    print("maxlen",maxlen)
    
    #加0补齐
    print("type_vdata",type(x_train[1][1][1]))
    x_train=vseq_padding(maxlen,x_train)
    x_test=vseq_padding(maxlen,x_test)
    x_train_un=seq_padding(maxlen,x_train_un)
    x_test_un=seq_padding(maxlen,x_test_un)
    x_test_un=np.reshape(x_test_un,(10000,maxlen))
    print("shape_x_train",np.shape(x_train[1]))
    
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

    x1=Lambda(spilt_4,output_shape=(None,word_size),arguments={'index':1})(x)
    mask1=Lambda(spilt_4,output_shape=(None,),arguments={'index':1})(mask)

    x2=Lambda(spilt_4,output_shape=(None,word_size),arguments={'index':2})(x)
    mask2=Lambda(spilt_4,output_shape=(None,),arguments={'index':2})(mask)

    x3=Lambda(spilt_4,output_shape=(None,word_size),arguments={'index':3})(x)
    mask3=Lambda(spilt_4,output_shape=(None,),arguments={'index':3})(mask)

    x4=Lambda(spilt_4,output_shape=(None,word_size),arguments={'index':4})(x)
    mask4=Lambda(spilt_4,output_shape=(None,),arguments={'index':4})(mask)


    x1 = DilatedGatedConv1D(rate=1, drop_gate=0.1)([x1, mask1])
    x1 = DilatedGatedConv1D(rate=2, drop_gate=0.2)([x1, mask1])
    x1 = DilatedGatedConv1D(rate=1, drop_gate=0.1)([x1, mask1])
    x1=Conv1D(250,5, activation='relu',padding='same')(x1)
    x1=Dropout(0.1)(x1)
    x1=Lambda(mask_ed)([x1,mask1])
    x1=Conv1D(270,5, activation='relu',padding='same')(x1)
    x1=Dropout(0.1)(x1)
    x1=Lambda(mask_ed)([x1,mask1])
    x1=Conv1D(290,5, activation='relu',padding='same')(x1)
    x1=Dropout(0.1)(x1)
    x1=Lambda(mask_ed)([x1,mask1])
    x1=Dense(200)(x1)
    x1=Dropout(0.1)(x1)
    x1=Lambda(mask_ed)([x1,mask1])
    x1=Dense(150)(x1)
    x1=Dropout(0.1)(x1)
    x1=Lambda(mask_ed)([x1,mask1])
    x1=Dense(130)(x1)
    x1=Dropout(0.1)(x1)
    x1=Lambda(mask_ed)([x1,mask1])
    x1=Dense(65)(x1)
    x1=Dropout(0.1)(x1)
    x1=Lambda(mask_ed)([x1,mask1])
    x1=Dense(40)(x1)
    x1=Dropout(0.1)(x1)
    """
    x1=Dense(32)(x1)
    x1=Dropout(0.1)(x1)
    x1=Dense(2)(x1)
    """


    x2 = DilatedGatedConv1D(rate=1, drop_gate=0.1)([x2, mask2])
    x2 = DilatedGatedConv1D(rate=2, drop_gate=0.2)([x2, mask2])
    x2 = DilatedGatedConv1D(rate=1, drop_gate=0.2)([x2, mask2])
    x2=Conv1D(250,5, activation='relu',padding='same')(x2)
    x2=Dropout(0.1)(x2)
    x2=Lambda(mask_ed)([x2,mask2])
    x2=Conv1D(270,5, activation='relu',padding='same')(x2)
    x2=Dropout(0.1)(x2)
    x2=Lambda(mask_ed)([x2,mask2])
    x2=Conv1D(290,5, activation='relu',padding='same')(x2)
    x1=Dropout(0.1)(x1)
    x2=Lambda(mask_ed)([x2,mask2])
    x2=Dense(200)(x2)
    x2=Dropout(0.1)(x2)
    x2=Lambda(mask_ed)([x2,mask2])
    x2=Dense(150)(x2)
    x2=Dropout(0.1)(x2)
    x2=Lambda(mask_ed)([x2,mask2])
    x2=Dense(130)(x2)
    x2=Dropout(0.1)(x2)
    x2=Lambda(mask_ed)([x2,mask2])
    x2=Dense(65)(x2)
    x2=Dropout(0.1)(x2)
    x2=Lambda(mask_ed)([x2,mask2])
    x2=Dense(40)(x2)
    x2=Dropout(0.1)(x2)
    """
    x2=Dense(32)(x2)
    x2=Dropout(0.1)(x2)
    x2=Dense(2)(x2)
    """

    x3 = DilatedGatedConv1D(rate=1, drop_gate=0.1)([x3, mask3])
    x3 = DilatedGatedConv1D(rate=2, drop_gate=0.2)([x3, mask3])
    x3 = DilatedGatedConv1D(rate=1, drop_gate=0.4)([x3, mask3])
    x3=Conv1D(filters=230,kernel_size=5, activation='relu',padding='same')(x3)
    x3=Dropout(0.1)(x3)
    x3=Lambda(mask_ed)([x3,mask3])
    x3=Conv1D(220,5, activation='relu',padding='same')(x3)
    x3=Dropout(0.2)(x3)
    x3=Lambda(mask_ed)([x3,mask3])
    x3=Conv1D(210,5, activation='relu',padding='same')(x3)
    x3=Dropout(0.3)(x3)
    x3=Lambda(mask_ed)([x3,mask3])
    x3=Dense(200)(x3)
    x3=Dropout(0.3)(x3)
    x3=Lambda(mask_ed)([x3,mask3])
    x3=Dense(150)(x3)
    x3=Dropout(0.2)(x3)
    x3=Lambda(mask_ed)([x3,mask3])
    x3=Dense(130)(x3)
    x3=Dropout(0.3)(x3)
    x3=Lambda(mask_ed)([x3,mask3])
    x3=Dense(65)(x3)
    x3=Dropout(0.2)(x3)
    x3=Lambda(mask_ed)([x3,mask3])
    x3=Dense(40)(x3)
    x3=Dropout(0.2)(x3)
    """
    x3=Dense(32)(x3)
    x3=Dropout(0.1)(x3)
    x3=Dense(2)(x3)
    """

    x4 = DilatedGatedConv1D(rate=1, drop_gate=0.2)([x4, mask4])
    x4 = DilatedGatedConv1D(rate=2, drop_gate=0.4)([x4, mask4])
    x4 = DilatedGatedConv1D(rate=1, drop_gate=0.6)([x4, mask4])
    x4=Conv1D(220,5, activation='relu',padding='same')(x4)
    x4=Dropout(0.2)(x4)
    x4=Lambda(mask_ed)([x4,mask4])
    x4=Conv1D(210,5, activation='relu',padding='same')(x4)
    x4=Dropout(0.4)(x4)
    x4=Lambda(mask_ed)([x4,mask4])
    x4=Conv1D(200,5, activation='relu',padding='same')(x4)
    x4=Dropout(0.6)(x4)
    x4=Lambda(mask_ed)([x4,mask4])
    x4=Dense(150)(x4)
    x4=Dropout(0.5)(x4)
    x4=Lambda(mask_ed)([x4,mask4])
    x4=Dense(130)(x4)
    x4=Dropout(0.6)(x4)
    x4=Lambda(mask_ed)([x4,mask4])
    x4=Dense(65)(x4)
    x4=Dropout(0.4)(x4)
    x4=Lambda(mask_ed)([x4,mask4])
    x4=Dense(40)(x4)
    x4=Dropout(0.5)(x4)
    """
    x4=Dense(32)(x4)
    x4=Dropout(0.1)(x4)
    x4=Dense(2)(x4)
    """
    x=Lambda(union_4)([x1,x2,x3,x4])
    x=Lambda(mask_ed)([x,mask])
    x=Dense(80,activation='relu')(x)
    x=Dropout(0.1)(x)
    x=Lambda(mask_ed)([x,mask])
    x=Dense(40,activation='relu')(x)
    x=Dropout(0.1)(x)
    x=Lambda(mask_ed)([x,mask])
    x=Dense(20,activation='relu')(x)
    x=Dropout(0.1)(x)
    x=Lambda(mask_ed)([x,mask])
    x=Dense(2,activation='relu')(x)
    x=Dropout(0.1)(x)
    x=Lambda(mask_ed)([x,mask])
    p = Dense(1, activation='sigmoid')(x)
    model=Model([x_in,x_in_un],p)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    return model


if __name__ == "__main__":
    x_train,x_train_un,y_train,x_test,x_test_un,y_test,maxlen,word_size=data_set()
    model=build_model(maxlen,word_size)
    history=model.fit([x_train,x_train_un],y_train,epochs=20,batch_size=64)
    loss,accuracy=model.evaluate([x_test,x_test_un],y_test)
    print(loss,accuracy)