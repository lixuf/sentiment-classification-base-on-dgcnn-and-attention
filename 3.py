
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
from keras.preprocessing import sequence
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
import tensorflow as tf
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
    return K.concatenate([x1,x2,x3,x4],axis=-1)

def mask_ed(x):
    x,mask=x
    return x*mask


def word2vec(in_txt):
    path = get_tmpfile("word2vec_lstm.model")
    model = Word2Vec(in_txt, size=150, window=6, min_count=1,
                     workers=multiprocessing.cpu_count())
    model.save("word2vec_lstm.model")

def data_set():
    data_vol=15000
    (x_train_in, y_train), (x_test_in, y_test) = imdb.load_data()
    print("x_train",len(x_train_in))
    print("x_test",len(x_test_in))
    x_train_in=x_train_in[:data_vol]
    x_test_in=x_test_in[:data_vol]
    y_train=y_train[:data_vol]
    y_test=y_test[:data_vol]
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

    y_train=np.reshape(y_train,(data_vol,1))
    y_test=np.reshape(y_test,(data_vol,1))
    print("shape_y_train",np.shape(y_train))
    print("maxlen",maxlen)
    
    #加0补齐
    print("type_vdata",type(x_train[1][1][1]))
    x_train=sequence.pad_sequences(x_train,maxlen=maxlen,padding='post',truncating="post")
    x_test=sequence.pad_sequences(x_test,maxlen=maxlen,padding='post',truncating="post")
    x_train_un=sequence.pad_sequences(x_train_un,maxlen=maxlen,padding='post',truncating="post")
    x_test_un=sequence.pad_sequences(x_test_un,maxlen=maxlen,padding='post',truncating="post")
    x_test_un=np.reshape(x_test_un,(data_vol,maxlen))
    x_train_un=np.reshape(x_train_un,(data_vol,maxlen))

    print("shape_x_train",np.shape(x_train))
    
    print("shape_x_train_un",np.shape(x_train_un))
  
    print("len_words",len(x_train[1]))
    print("len_words",len(x_test[1]))
    word_size=150
    return x_train,x_train_un,y_train,x_test,x_test_un,y_test,maxlen,word_size,data_vol

def block_dgcnn(x,mask,dro):

    x = DilatedGatedConv1D(rate=1, drop_gate=0.2)([x, mask])
    x = Dense(50,activation='relu')(x)
    x = Dropout(dro)(x)
    x=GlobalMaxPooling1D()(x)
    return x

def block_cnn(x,mask,dro):
    x = Conv1D(175,kernel_size=3,padding='same', activation='relu')(x)
    x = Dropout(dro)(x)
    x = Dense(50,activation='relu')(x)
    x = Dropout(dro)(x)
    x= GlobalMaxPooling1D()(x)
    return x

def pad_re(x,maxlen,batch_size,n):
    x,x_re=x
    n1=n
    for i in range(batch_size):
        while n[i]<maxlen:
            mid=int(maxlen-n[i])
            if mid<=n1[i]:
                x=tf.assign=(x[i][int(n[i]):],x_re[i][:mid])
                n[i]=maxlen
            else:
                x=tf.assign(x[i][int(n[i]):int(n[i])+int(n1[i])],x_re[i][:int(n1[i])])
                n[i]=n[i]+n1[i]

    return x

def seq_len_f(x):
    seq_len = K.round(K.sum(x, 1)[:, 0])
    seq_len = K.cast(seq_len, 'int32')
    return seq_len

def reverse_sequence(x):
    x,mask,seq_len=x
    return tf.reverse_sequence(x, seq_len, seq_dim=1)

def build_model(maxlen,word_size,data_vol,batch_size):
    x_in= Input(shape=(None,word_size))
    x_in_un = Input(shape=(None,))
    pid = Lambda(position_id)(x_in_un)
    position_embedding = Embedding(maxlen, word_size, embeddings_initializer='zeros')
    pv = position_embedding(pid)
    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x_in_un)
    x=Dense(word_size,use_bias=False)(x_in)
    x = Add()([x, pv])

    n=Lambda(seq_len_f,output_shape=(batch_size,1))(mask)
    x_re=Lambda(reverse_sequence,output_shape=(None,word_size))([x,mask,n])#转向
    x=Lambda(pad_re,output_shape=(None,word_size),arguments={'maxlen':maxlen,'batch_size':batch_size,'n':K.eval(n)})([x,x_re])#将转向填充进去

    x1=Lambda(spilt_4,output_shape=(None,word_size),arguments={'index':1})(x)
    mask1=Lambda(spilt_4,output_shape=(None,),arguments={'index':1})(mask)

    x2=Lambda(spilt_4,output_shape=(None,word_size),arguments={'index':2})(x)
    mask2=Lambda(spilt_4,output_shape=(None,),arguments={'index':2})(mask)

    x3=Lambda(spilt_4,output_shape=(None,word_size),arguments={'index':3})(x)
    mask3=Lambda(spilt_4,output_shape=(None,),arguments={'index':3})(mask)

    x4=Lambda(spilt_4,output_shape=(None,word_size),arguments={'index':4})(x)
    mask4=Lambda(spilt_4,output_shape=(None,),arguments={'index':4})(mask)

   

    x1 = block_cnn(x1,mask1,0.1)

    x2 = block_cnn(x2,mask2,0.2)

    x3 = block_cnn(x3,mask3,0.3)

    x4 = block_cnn(x4,mask4,0.4)
 

    x=Lambda(union_4)([x1,x2,x3,x4])
    p = Dense(1, activation='sigmoid')(x)
    model=Model([x_in,x_in_un],p)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    return model


if __name__ == "__main__":
    x_train,x_train_un,y_train,x_test,x_test_un,y_test,maxlen,word_size,data_vol=data_set()
    batch_size=64
    model=build_model(maxlen,word_size,data_vol,batch_size)
    history=model.fit([x_train,x_train_un],y_train,epochs=8,batch_size=batch_size)
    loss,accuracy=model.evaluate([x_test,x_test_un],y_test)
    print(loss,accuracy)