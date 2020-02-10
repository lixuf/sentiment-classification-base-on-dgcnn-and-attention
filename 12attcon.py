






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
from keras.callbacks import Callback
import tensorflow as tf

"""
回调器
"""
from keras.callbacks import Callback

def evaluate(): # 评测函数
    pred = model.predict([x_train,x_train_un])
    return np.mean(pred.argmax(axis=1) == y_train) # 爱算啥就算啥


# 定义Callback器，计算验证集的acc，并保存最优模型
class Evaluate(Callback):

    def __init__(self):
        self.accs = []
        self.highest = 0.


    def on_epoch_end(self, epoch, logs=None):
        loss,acc=model.evaluate([x_test,x_test_un],y_test)
        self.accs.append(acc)
    
        if acc >= self.highest: # 保存最优模型权重
            self.highest = acc
            model.save_weights('best_model.weights')
        
        # 爱运行什么就运行什么
        print ("acc:", acc,"loss:",loss," highest:",self.highest ) 

"""
"""

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


class OurBidirectional(OurLayer):

    def __init__(self, layer, **args):
        super(OurBidirectional, self).__init__(**args)
        self.forward_layer = layer.__class__.from_config(layer.get_config())
        self.backward_layer = layer.__class__.from_config(layer.get_config())
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name
    def reverse_sequence(self, x, mask):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return tf.reverse_sequence(x, seq_len, seq_dim=1)
    def call(self, inputs):
        x, mask = inputs
        x_forward = self.reuse(self.forward_layer, [x,mask])
        x_backward = self.reverse_sequence(x, mask)
        x_backward = self.reuse(self.backward_layer, [x_backward,mask])
        x_backward = self.reverse_sequence(x_backward, mask)
        x = (x_forward+x_backward)/2
        if K.ndim(x) == 3:
            return x * mask
        else:
            return x
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.forward_layer.o_dim,)

class AttentionConatenate(OurLayer):
    
    def __init__(self, h_dim, **kwargs):
        super(AttentionConatenate, self).__init__(**kwargs)
        self.h_dim = h_dim
    def build(self, input_shape):
        super(AttentionConatenate, self).build(input_shape)
   
        self.k1_dense = Dense(
            self.h_dim,
            use_bias=False,
            activation='tanh'
        )
        self.o1_dense = Dense(1, use_bias=False)

        self.k2_dense = Dense(
            self.h_dim,
            use_bias=False,
            activation='tanh'
        )
        self.o2_dense = Dense(1, use_bias=False)

        self.k3_dense = Dense(
            self.h_dim,
            use_bias=False,
            activation='tanh'
        )
        self.o3_dense = Dense(1, use_bias=False)

        self.k4_dense = Dense(
            self.h_dim,
            use_bias=False,
            activation='tanh'
        )
        self.o4_dense = Dense(1, use_bias=False)
    def call(self, inputs):
        x1,x2,x3,x4 = inputs
        x=K.concatenate([K.expand_dims(x1, axis=-2),K.expand_dims(x2, axis=-2),K.expand_dims(x3, axis=-2),K.expand_dims(x4, axis=-2)],axis=-2)
        x1 = self.reuse(self.k1_dense, x1)
        x1 = self.reuse(self.o1_dense, x1)
        x2 = self.reuse(self.k2_dense, x2)
        x2 = self.reuse(self.o2_dense, x2)
        x3 = self.reuse(self.k3_dense, x3)
        x3 = self.reuse(self.o3_dense, x3)
        x4 = self.reuse(self.k4_dense, x4)
        x4 = self.reuse(self.o4_dense, x4)
        xu = K.concatenate([K.expand_dims(x1, axis=-2),K.expand_dims(x2, axis=-2),K.expand_dims(x3, axis=-2),K.expand_dims(x4, axis=-2)],axis=-2)
        xu = K.softmax(xu, -2)
        return K.sum(x * xu,1)
    def compute_output_shape(self, input_shape):
        return (None, self.h_dim)

class AttentionPooling1D(OurLayer):
    """通过加性Attention，将向量序列融合为一个定长向量
    """
    def __init__(self, h_dim=None, **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.h_dim = h_dim
    def build(self, input_shape):
        super(AttentionPooling1D, self).build(input_shape)
        if self.h_dim is None:
            self.h_dim = input_shape[0][-1]
        self.k_dense = Dense(
            self.h_dim,
            use_bias=False,
            activation='tanh'
        )
        self.o_dense = Dense(1, use_bias=False)
    def call(self, inputs):
        xo, mask = inputs
        x = xo
        x = self.reuse(self.k_dense, x)
        x = self.reuse(self.o_dense, x)
        x = x - (1 - mask) * 1e12
        x = K.softmax(x, 1)
        return K.sum(x * xo, 1)
    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][-1])

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
                senten_un.append(x_test_in_int[i][i2])
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
    x_train=sequence.pad_sequences(x_train,maxlen=maxlen,dtype="float32",padding='post',truncating="post")
    x_test=sequence.pad_sequences(x_test,maxlen=maxlen,dtype="float32",padding='post',truncating="post")
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

    x1 = OurBidirectional(DilatedGatedConv1D(rate=1, drop_gate=dro))([x, mask])
    x1 = OurBidirectional(DilatedGatedConv1D(rate=2, drop_gate=dro))([x1, mask])
    x1 = OurBidirectional(DilatedGatedConv1D(rate=4, drop_gate=dro))([x1, mask])
    x1 = OurBidirectional(DilatedGatedConv1D(rate=1, drop_gate=dro))([x1, mask])
    x=Add()([x,x1])
    x=AttentionPooling1D()([x,mask])
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


def build_model_dgcnn(maxlen,word_size,data_vol,batch_size):
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

   

    x1 = block_dgcnn(x1,mask1,0.1)


    x2 = block_dgcnn(x2,mask2,0.2)
 

    x3 = block_dgcnn(x3,mask3,0.3)
  

    x4 = block_dgcnn(x4,mask4,0.4)
 
 
    x=AttentionConatenate(150)([x1,x2,x3,x4])

    p = Dense(1, activation='sigmoid')(x)
    model=Model([x_in,x_in_un],p)
    
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    return model



if __name__ == "__main__":
    x_train,x_train_un,y_train,x_test,x_test_un,y_test,maxlen,word_size,data_vol=data_set()
    batch_size=64
    model=build_model_dgcnn(maxlen,word_size,data_vol,batch_size)
    evaluator = Evaluate()
    history=model.fit([x_train,x_train_un],y_train,epochs=50,batch_size=batch_size,callbacks=[evaluator]) 
    print(evaluator.accs)



"""
library cublas64_100.dll
15000/15000 [==============================] - 85s 6ms/step - loss: 0.4466 - acc: 0.7911
15000/15000 [==============================] - 68s 5ms/step
acc: 0.8368666666666666 loss: 0.37321069496472675  highest: 0.8368666666666666
Epoch 2/50
15000/15000 [==============================] - 102s 7ms/step - loss: 0.3571 - acc: 0.8426
15000/15000 [==============================] - 81s 5ms/step
acc: 0.8492 loss: 0.34151815980275474  highest: 0.8492
Epoch 3/50
15000/15000 [==============================] - 88s 6ms/step - loss: 0.3337 - acc: 0.8575
15000/15000 [==============================] - 77s 5ms/step
acc: 0.8552 loss: 0.3325901741027832  highest: 0.8552
Epoch 4/50
15000/15000 [==============================] - 85s 6ms/step - loss: 0.3192 - acc: 0.8633
15000/15000 [==============================] - 71s 5ms/step
acc: 0.8628 loss: 0.31887869209448494  highest: 0.8628
Epoch 5/50
15000/15000 [==============================] - 82s 5ms/step - loss: 0.3060 - acc: 0.8707
15000/15000 [==============================] - 70s 5ms/step
acc: 0.8672 loss: 0.31400098497867585  highest: 0.8672
Epoch 6/50
15000/15000 [==============================] - 86s 6ms/step - loss: 0.2992 - acc: 0.8739
15000/15000 [==============================] - 68s 5ms/step
acc: 0.8688666666666667 loss: 0.3094554013490677  highest: 0.8688666666666667
Epoch 7/50
15000/15000 [==============================] - 87s 6ms/step - loss: 0.2886 - acc: 0.8778
15000/15000 [==============================] - 67s 4ms/step
acc: 0.8706 loss: 0.30692052875359854  highest: 0.8706
Epoch 8/50
15000/15000 [==============================] - 79s 5ms/step - loss: 0.2804 - acc: 0.8817
15000/15000 [==============================] - 67s 4ms/step
acc: 0.8697333333651225 loss: 0.30939636261463166  highest: 0.8706
Epoch 9/50
15000/15000 [==============================] - 82s 5ms/step - loss: 0.2758 - acc: 0.8867
15000/15000 [==============================] - 80s 5ms/step
acc: 0.8671333333333333 loss: 0.3127993511676788  highest: 0.8706
Epoch 10/50
15000/15000 [==============================] - 86s 6ms/step - loss: 0.2677 - acc: 0.8881
15000/15000 [==============================] - 84s 6ms/step
acc: 0.8674666666984558 loss: 0.3126197373787562  highest: 0.8706
Epoch 11/50
15000/15000 [==============================] - 81s 5ms/step - loss: 0.2666 - acc: 0.8871
15000/15000 [==============================] - 69s 5ms/step
acc: 0.8632 loss: 0.32140862289269767  highest: 0.8706
Epoch 12/50
15000/15000 [==============================] - 80s 5ms/step - loss: 0.2591 - acc: 0.8904
15000/15000 [==============================] - 76s 5ms/step
acc: 0.8703333333333333 loss: 0.30978180900414787  highest: 0.8706
Epoch 13/50
15000/15000 [==============================] - 89s 6ms/step - loss: 0.2502 - acc: 0.8959
15000/15000 [==============================] - 72s 5ms/step
acc: 0.8676666666666667 loss: 0.3218943000396093  highest: 0.8706
"""

"""
library cublas64_100.dll
15000/15000 [==============================] - 104s 7ms/step - loss: 0.4289 - acc: 0.8012
15000/15000 [==============================] - 72s 5ms/step
acc: 0.8463333333333334 loss: 0.3601866277853648  highest: 0.8463333333333334
Epoch 2/50
15000/15000 [==============================] - 90s 6ms/step - loss: 0.3466 - acc: 0.8499
15000/15000 [==============================] - 70s 5ms/step
acc: 0.8508 loss: 0.3367941662947337  highest: 0.8508
Epoch 3/50
15000/15000 [==============================] - 83s 6ms/step - loss: 0.3275 - acc: 0.8591
15000/15000 [==============================] - 76s 5ms/step
acc: 0.8449999999682108 loss: 0.3478331519762675  highest: 0.8508
Epoch 4/50
15000/15000 [==============================] - 90s 6ms/step - loss: 0.3157 - acc: 0.8655
15000/15000 [==============================] - 74s 5ms/step
acc: 0.8618000000317891 loss: 0.31859110669294993  highest: 0.8618000000317891
Epoch 5/50
15000/15000 [==============================] - 90s 6ms/step - loss: 0.3040 - acc: 0.8729
15000/15000 [==============================] - 83s 6ms/step
acc: 0.8672 loss: 0.31362033146222434  highest: 0.8672
Epoch 6/50
15000/15000 [==============================] - 84s 6ms/step - loss: 0.2955 - acc: 0.8743
15000/15000 [==============================] - 80s 5ms/step
acc: 0.8708666666666667 loss: 0.3045446805238724  highest: 0.8708666666666667
Epoch 7/50
15000/15000 [==============================] - 82s 5ms/step - loss: 0.2874 - acc: 0.8790
15000/15000 [==============================] - 72s 5ms/step
acc: 0.8633999999682108 loss: 0.31491973698933917  highest: 0.8708666666666667
Epoch 8/50
15000/15000 [==============================] - 97s 6ms/step - loss: 0.2785 - acc: 0.8824
15000/15000 [==============================] - 69s 5ms/step
acc: 0.8722666666666666 loss: 0.3087876972754796  highest: 0.8722666666666666
Epoch 9/50
15000/15000 [==============================] - 85s 6ms/step - loss: 0.2742 - acc: 0.8847
15000/15000 [==============================] - 75s 5ms/step
acc: 0.868 loss: 0.3120438311576843  highest: 0.8722666666666666
Epoch 10/50
15000/15000 [==============================] - 108s 7ms/step - loss: 0.2673 - acc: 0.8881
15000/15000 [==============================] - 75s 5ms/step
acc: 0.8712666666984558 loss: 0.30820263817310334  highest: 0.8722666666666666
Epoch 11/50
15000/15000 [==============================] - 97s 6ms/step - loss: 0.2666 - acc: 0.8891
15000/15000 [==============================] - 76s 5ms/step
acc: 0.8632000000317891 loss: 0.32060642166932424  highest: 0.8722666666666666
Epoch 12/50
15000/15000 [==============================] - 85s 6ms/step - loss: 0.2590 - acc: 0.8892
15000/15000 [==============================] - 65s 4ms/step
acc: 0.8698 loss: 0.31082726918061576  highest: 0.8722666666666666
Epoch 13/50
15000/15000 [==============================] - 91s 6ms/step - loss: 0.2500 - acc: 0.8971
15000/15000 [==============================] - 78s 5ms/step
acc: 0.8659333333333333 loss: 0.32143214031060535  highest: 0.8722666666666666
Epoch 14/50
15000/15000 [==============================] - 84s 6ms/step - loss: 0.2465 - acc: 0.8976
15000/15000 [==============================] - 77s 5ms/step
acc: 0.8672666666666666 loss: 0.3177384898106257  highest: 0.8722666666666666
Epoch 15/50
15000/15000 [==============================] - 94s 6ms/step - loss: 0.2427 - acc: 0.8987
15000/15000 [==============================] - 81s 5ms/step
acc: 0.8686 loss: 0.32432410172621406  highest: 0.8722666666666666
Epoch 16/50
15000/15000 [==============================] - 67s 4ms/step - loss: 0.2352 - acc: 0.9046
15000/15000 [==============================] - 55s 4ms/step
acc: 0.8622666666984559 loss: 0.32599000594615934  highest: 0.8722666666666666
Epoch 17/50
15000/15000 [==============================] - 60s 4ms/step - loss: 0.2269 - acc: 0.9059
15000/15000 [==============================] - 67s 4ms/step
acc: 0.8605333333333334 loss: 0.341108451739947  highest: 0.8722666666666666
Epoch 18/50
15000/15000 [==============================] - 62s 4ms/step - loss: 0.2160 - acc: 0.9129
15000/15000 [==============================] - 48s 3ms/step
acc: 0.8654666666666667 loss: 0.3414358741521835  highest: 0.8722666666666666
Epoch 19/50
15000/15000 [==============================] - 65s 4ms/step - loss: 0.2091 - acc: 0.9146
15000/15000 [==============================] - 61s 4ms/step
acc: 0.8586666666666667 loss: 0.34787100009918215  highest: 0.8722666666666666
Epoch 20/50
15000/15000 [==============================] - 65s 4ms/step - loss: 0.2081 - acc: 0.9153
15000/15000 [==============================] - 59s 4ms/step
acc: 0.8547333333651225 loss: 0.3744520626068115  highest: 0.8722666666666666
Epoch 21/50
15000/15000 [==============================] - 64s 4ms/step - loss: 0.1948 - acc: 0.9214
15000/15000 [==============================] - 69s 5ms/step
acc: 0.8536000000317892 loss: 0.3821171977996826  highest: 0.8722666666666666
Epoch 22/50
15000/15000 [==============================] - 71s 5ms/step - loss: 0.1856 - acc: 0.9283
15000/15000 [==============================] - 56s 4ms/step
acc: 0.8512666666666666 loss: 0.3913412213842074  highest: 0.8722666666666666
Epoch 23/50
15000/15000 [==============================] - 70s 5ms/step - loss: 0.1802 - acc: 0.9288
15000/15000 [==============================] - 57s 4ms/step
acc: 0.8405333333015442 loss: 0.4389837670485179  highest: 0.8722666666666666
Epoch 24/50
15000/15000 [==============================] - 62s 4ms/step - loss: 0.1747 - acc: 0.9308
15000/15000 [==============================] - 54s 4ms/step
acc: 0.8564 loss: 0.3907109208504359  highest: 0.8722666666666666
Epoch 25/50
15000/15000 [==============================] - 71s 5ms/step - loss: 0.1669 - acc: 0.9321
15000/15000 [==============================] - 59s 4ms/step
acc: 0.8535999999682109 loss: 0.4135424822727839  highest: 0.8722666666666666
Epoch 26/50
15000/15000 [==============================] - 62s 4ms/step - loss: 0.1564 - acc: 0.9394
15000/15000 [==============================] - 49s 3ms/step
acc: 0.8556000000317892 loss: 0.41815085143248243  highest: 0.8722666666666666
Epoch 27/50
15000/15000 [==============================] - 65s 4ms/step - loss: 0.1515 - acc: 0.9412
15000/15000 [==============================] - 50s 3ms/step
acc: 0.85 loss: 0.44919757029215496  highest: 0.8722666666666666
Epoch 28/50
15000/15000 [==============================] - 66s 4ms/step - loss: 0.1326 - acc: 0.9483
15000/15000 [==============================] - 55s 4ms/step
acc: 0.8495999999682109 loss: 0.4540696466604869  highest: 0.8722666666666666
Epoch 29/50
15000/15000 [==============================] - 66s 4ms/step - loss: 0.1322 - acc: 0.9467
15000/15000 [==============================] - 66s 4ms/step
acc: 0.8467333333015442 loss: 0.4931726202170054  highest: 0.8722666666666666
Epoch 30/50
15000/15000 [==============================] - 71s 5ms/step - loss: 0.1214 - acc: 0.9539
15000/15000 [==============================] - 55s 4ms/step
acc: 0.8525999999682109 loss: 0.4956745819747448  highest: 0.8722666666666666
Epoch 31/50
15000/15000 [==============================] - 70s 5ms/step - loss: 0.1093 - acc: 0.9597
15000/15000 [==============================] - 63s 4ms/step
acc: 0.8456666666984558 loss: 0.5009315987785657  highest: 0.8722666666666666
Epoch 32/50
15000/15000 [==============================] - 65s 4ms/step - loss: 0.0979 - acc: 0.9637
15000/15000 [==============================] - 51s 3ms/step
acc: 0.8450666666984558 loss: 0.5371953658302625  highest: 0.8722666666666666
Epoch 33/50
15000/15000 [==============================] - 74s 5ms/step - loss: 0.1060 - acc: 0.9596
15000/15000 [==============================] - 58s 4ms/step
acc: 0.8421999999682108 loss: 0.5709907899975777  highest: 0.8722666666666666
Epoch 34/50
15000/15000 [==============================] - 70s 5ms/step - loss: 0.0841 - acc: 0.9679
15000/15000 [==============================] - 62s 4ms/step
acc: 0.8439333333651224 loss: 0.5931693685889244  highest: 0.8722666666666666
Epoch 35/50
15000/15000 [==============================] - 72s 5ms/step - loss: 0.0910 - acc: 0.9655
15000/15000 [==============================] - 59s 4ms/step
acc: 0.8474666666984558 loss: 0.5741449032704036  highest: 0.8722666666666666
Epoch 36/50
15000/15000 [==============================] - 68s 5ms/step - loss: 0.0837 - acc: 0.9670
15000/15000 [==============================] - 54s 4ms/step
acc: 0.8473333333015441 loss: 0.5864267010728518  highest: 0.8722666666666666
Epoch 37/50
15000/15000 [==============================] - 65s 4ms/step - loss: 0.0753 - acc: 0.9725
15000/15000 [==============================] - 55s 4ms/step
acc: 0.8354000000317892 loss: 0.656713459054629  highest: 0.8722666666666666
Epoch 38/50
15000/15000 [==============================] - 71s 5ms/step - loss: 0.0727 - acc: 0.9743
15000/15000 [==============================] - 69s 5ms/step
acc: 0.8451333333015442 loss: 0.6377758503874142  highest: 0.8722666666666666
Epoch 39/50
15000/15000 [==============================] - 74s 5ms/step - loss: 0.0651 - acc: 0.9754
15000/15000 [==============================] - 55s 4ms/step
acc: 0.8321333333333333 loss: 0.6901388391017914  highest: 0.8722666666666666
Epoch 40/50
15000/15000 [==============================] - 63s 4ms/step - loss: 0.0608 - acc: 0.9788
15000/15000 [==============================] - 47s 3ms/step
acc: 0.8409333333015442 loss: 0.6853919173399607  highest: 0.8722666666666666
Epoch 41/50
15000/15000 [==============================] - 94s 6ms/step - loss: 0.0634 - acc: 0.9763
15000/15000 [==============================] - 55s 4ms/step
acc: 0.8315333333333333 loss: 0.7408475859244664  highest: 0.8722666666666666
Epoch 42/50
15000/15000 [==============================] - 85s 6ms/step - loss: 0.0539 - acc: 0.9809
15000/15000 [==============================] - 65s 4ms/step
acc: 0.8388 loss: 0.7421364627679189  highest: 0.8722666666666666
Epoch 43/50
15000/15000 [==============================] - 67s 4ms/step - loss: 0.0554 - acc: 0.9797
15000/15000 [==============================] - 60s 4ms/step
acc: 0.8317999999682109 loss: 0.7737306351780892  highest: 0.8722666666666666
Epoch 44/50
15000/15000 [==============================] - 66s 4ms/step - loss: 0.0738 - acc: 0.9707
15000/15000 [==============================] - 58s 4ms/step
acc: 0.8428000000317891 loss: 0.7184555052479108  highest: 0.8722666666666666
Epoch 45/50
15000/15000 [==============================] - 72s 5ms/step - loss: 0.0484 - acc: 0.9837
15000/15000 [==============================] - 59s 4ms/step
acc: 0.8403333333015441 loss: 0.7754311492780844  highest: 0.8722666666666666
Epoch 46/50
15000/15000 [==============================] - 76s 5ms/step - loss: 0.0322 - acc: 0.9893
15000/15000 [==============================] - 60s 4ms/step
acc: 0.8368666666348775 loss: 0.7985089233160019  highest: 0.8722666666666666
Epoch 47/50
15000/15000 [==============================] - 68s 5ms/step - loss: 0.0480 - acc: 0.9835
15000/15000 [==============================] - 70s 5ms/step
acc: 0.8319333333015442 loss: 0.8877185416956742  highest: 0.8722666666666666
Epoch 48/50
15000/15000 [==============================] - 71s 5ms/step - loss: 0.0612 - acc: 0.9779
15000/15000 [==============================] - 47s 3ms/step
acc: 0.8339333333015442 loss: 0.8152066183805465  highest: 0.8722666666666666
Epoch 49/50
15000/15000 [==============================] - 71s 5ms/step - loss: 0.0459 - acc: 0.9829
15000/15000 [==============================] - 53s 4ms/step
acc: 0.8387999999682109 loss: 0.8363156028270722  highest: 0.8722666666666666
Epoch 50/50
15000/15000 [==============================] - 74s 5ms/step - loss: 0.0471 - acc: 0.9845
15000/15000 [==============================] - 55s 4ms/step
acc: 0.8404666666348776 loss: 0.8132609221458436  highest: 0.8722666666666666
[0.8463333333333334, 0.8508, 0.8449999999682108, 0.8618000000317891, 0.8672, 0.8708666666666667, 0.8633999999682108, 0.8722666666666666, 0.868, 0.8712666666984558, 0.8632000000317891, 0.8698, 0.8659333333333333, 0.8672666666666666, 0.8686, 0.8622666666984559, 0.8605333333333334, 0.8654666666666667, 0.8586666666666667, 0.8547333333651225, 0.8536000000317892, 0.8512666666666666, 0.8405333333015442, 0.8564, 0.8535999999682109, 0.8556000000317892, 0.85, 0.8495999999682109, 0.8467333333015442, 0.8525999999682109, 0.8456666666984558, 0.8450666666984558, 0.8421999999682108, 0.8439333333651224, 0.8474666666984558, 0.8473333333015441, 0.8354000000317892, 0.8451333333015442, 0.8321333333333333, 0.8409333333015442, 0.8315333333333333, 0.8388, 0.8317999999682109, 0.8428000000317891, 0.8403333333015441, 0.8368666666348775, 0.8319333333015442, 0.8339333333015442, 0.8387999999682109, 0.8404666666348776]

"""

"""
先pool 加残差
15000/15000 [==============================] - 250s 17ms/step - loss: 0.3882 - acc: 0.8289
15000/15000 [==============================] - 122s 8ms/step
acc: 0.8764666666666666 loss: 0.30082200264930725  highest: 0.8764666666666666
Epoch 2/50
15000/15000 [==============================] - 214s 14ms/step - loss: 0.2840 - acc: 0.8821
15000/15000 [==============================] - 97s 6ms/step
acc: 0.8802666666348775 loss: 0.28548635849952697  highest: 0.8802666666348775
Epoch 3/50
15000/15000 [==============================] - 196s 13ms/step - loss: 0.2290 - acc: 0.9055
15000/15000 [==============================] - 93s 6ms/step
acc: 0.8830666666984558 loss: 0.28270605573654173  highest: 0.8830666666984558
Epoch 4/50
15000/15000 [==============================] - 193s 13ms/step - loss: 0.1852 - acc: 0.9251
15000/15000 [==============================] - 116s 8ms/step
acc: 0.8805333333651225 loss: 0.31587538672288257  highest: 0.8830666666984558
Epoch 5/50
15000/15000 [==============================] - 204s 14ms/step - loss: 0.1299 - acc: 0.9467
15000/15000 [==============================] - 110s 7ms/step
acc: 0.8715333333651225 loss: 0.3499996879816055  highest: 0.8830666666984558
"""