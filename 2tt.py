
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
        self.pred=[]

    def on_epoch_end(self, epoch, logs=None):
        loss,acc=model.evaluate([x_test,x_test_un],y_test)
        self.accs.append(acc)
        self.pred=np.greater(model.predict([x_test,x_test_un]),0.5).astype(np.int16) 
    
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

class Attention(OurLayer):

    """多头注意力机制

    """

    def __init__(self, heads, size_per_head, key_size=None,

                 mask_right=False, **kwargs):

        super(Attention, self).__init__(**kwargs)

        self.heads = heads

        self.size_per_head = size_per_head

        self.out_dim = heads * size_per_head

        self.key_size = key_size if key_size else size_per_head

        self.mask_right = mask_right

    def build(self, input_shape):

        super(Attention, self).build(input_shape)

        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)

        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)

        self.v_dense = Dense(self.out_dim, use_bias=False)

    def call(self, inputs):

        q, k, v = inputs[: 3]

        v_mask, q_mask = None, None

        # 这里的mask.shape=[batch_size, seq_len]或[batch_size, seq_len, 1]

        if len(inputs) > 3:

            v_mask = inputs[3]

            if len(inputs) > 4:

                q_mask = inputs[4]

        # 线性变换

        qw = self.reuse(self.q_dense, q)

        kw = self.reuse(self.k_dense, k)

        vw = self.reuse(self.v_dense, v)

        # 形状变换

        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))

        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))

        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))

        # 维度置换

        qw = K.permute_dimensions(qw, (0, 2, 1, 3))

        kw = K.permute_dimensions(kw, (0, 2, 1, 3))

        vw = K.permute_dimensions(vw, (0, 2, 1, 3))

        # Attention

        a = K.batch_dot(qw, kw, [3, 3]) / self.key_size**0.5

        a = K.permute_dimensions(a, (0, 3, 2, 1))

        a = to_mask(a, v_mask, 'add')

        a = K.permute_dimensions(a, (0, 3, 2, 1))

        if (self.mask_right is not False) or (self.mask_right is not None):

            if self.mask_right is True:

                ones = K.ones_like(a[: 1, : 1])

                mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10

                a = a - mask

            else:

                # 这种情况下，mask_right是外部传入的0/1矩阵，shape=[q_len, k_len]

                mask = (1 - K.constant(self.mask_right)) * 1e10

                mask = K.expand_dims(K.expand_dims(mask, 0), 0)

                self.mask = mask

                a = a - mask

        a = K.softmax(a)

        self.a = a

        # 完成输出

        o = K.batch_dot(a, vw, [3, 2])

        o = K.permute_dimensions(o, (0, 2, 1, 3))

        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))

        o = to_mask(o, q_mask, 'mul')

        return o

    def compute_output_shape(self, input_shape):

        return (input_shape[0][0], input_shape[0][1], self.out_dim)


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


def to_mask(x, mask, mode='mul'):

    """通用mask函数

    这里的mask.shape=[batch_size, seq_len]或[batch_size, seq_len, 1]

    """

    if mask is None:

        return x

    else:

        for _ in range(K.ndim(x) - K.ndim(mask)):

            mask = K.expand_dims(mask, K.ndim(mask))

        if mode == 'mul':

            return x * mask

        else:

            return x - (1 - mask) * 1e10

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

    def call(self, inputs):
        x1,x2= inputs
        x=K.concatenate([K.expand_dims(x1, axis=-2),K.expand_dims(x2, axis=-2)],axis=-2)
        x1 = self.reuse(self.k1_dense, x1)
        x1 = self.reuse(self.o1_dense, x1)/self.h_dim**0.5
        x2 = self.reuse(self.k2_dense, x2)
        x2 = self.reuse(self.o2_dense, x2)/self.h_dim**0.5
        xu = K.concatenate([K.expand_dims(x1, axis=-2),K.expand_dims(x2, axis=-2)],axis=-2)
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
        x = self.reuse(self.o_dense, x)/self.h_dim**0.5
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
def con_res(x):
    x,y=x
    return K.concatenate([x,y],axis=-1)

def word2vec(in_txt):
    path = get_tmpfile("word2vec_lstm.model")
    model = Word2Vec(in_txt, size=150, window=6, min_count=1,
                     workers=multiprocessing.cpu_count())
    model.save("word2vec_lstm.model")

def data_set():
    data_vol=25000
    (x_train_in, y_train), (x_test_in, y_test) = imdb.load_data()
    print("x_train",len(x_train_in))
    print("x_test",len(x_test_in))
    x_train_in=x_train_in[:data_vol]
    x_test_in=x_test_in[:data_vol]
    y_train=y_train[:data_vol]
    y_test=y_test[:data_vol]
    x_train_in_int=x_train_in
    x_test_in_int=x_test_in
    maxlen=500
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
                print("wroud")
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

def block_dgcnn(x,mask,share,dro):

    x1 = share([x, mask])
    x1 = OurBidirectional(DilatedGatedConv1D(rate=2, drop_gate=dro))([x1, mask])
    x1 = OurBidirectional(DilatedGatedConv1D(rate=4, drop_gate=dro))([x1, mask])
    x1 = OurBidirectional(DilatedGatedConv1D(rate=1, drop_gate=dro))([x1, mask])
    x1 = Dense(75, use_bias=False)(x1)
    x1 = Lambda(mask_ed)([x1,mask])
    x  = Dense(75, use_bias=False)(x)
    x  = Lambda(mask_ed)([x,mask])
    x  = Lambda(con_res)([x,x1])
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



   
    share=OurBidirectional(DilatedGatedConv1D(rate=1, drop_gate=0.2))

    x1 = block_dgcnn(x1,mask1,share,0.1)


    x2 = block_dgcnn(x2,mask2,share,0.2)
 

 
 
    x=AttentionConatenate(150)([x1,x2])

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
    history=model.fit([x_train,x_train_un],y_train,epochs=3,batch_size=batch_size,callbacks=[evaluator]) 
    print(evaluator.accs)
    history=model.fit([x_test,x_test_un],evaluator.pred,epochs=10,batch_size=batch_size,callbacks=[evaluator])
    print(evaluator.accs)
"""
15000/15000 [==============================] - 243s 16ms/step - loss: 0.3989 - acc: 0.8163
15000/15000 [==============================] - 110s 7ms/step
acc: 0.8621333333651224 loss: 0.32768460791905724  highest: 0.8621333333651224
Epoch 2/3
15000/15000 [==============================] - 208s 14ms/step - loss: 0.2829 - acc: 0.8834
15000/15000 [==============================] - 98s 7ms/step
acc: 0.8819333333333333 loss: 0.2811084714253744  highest: 0.8819333333333333
Epoch 3/3
15000/15000 [==============================] - 200s 13ms/step - loss: 0.2268 - acc: 0.9067
15000/15000 [==============================] - 98s 7ms/step
acc: 0.9008666666666667 loss: 0.24206683236757914  highest: 0.9008666666666667
[0.8621333333651224, 0.8819333333333333, 0.9008666666666667]
Epoch 1/3
15000/15000 [==============================] - 186s 12ms/step - loss: 0.2417 - acc: 3.3333e-04
15000/15000 [==============================] - 85s 6ms/step
acc: 0.9012 loss: 0.24166555949846905  highest: 0.9012
Epoch 2/3
15000/15000 [==============================] - 187s 12ms/step - loss: 0.2393 - acc: 3.3333e-04
15000/15000 [==============================] - 89s 6ms/step
acc: 0.8999333333333334 loss: 0.24230838134288787  highest: 0.9012
Epoch 3/3
"""


"""
15000/15000 [==============================] - 252s 17ms/step - loss: 0.3989 - acc: 0.8163
15000/15000 [==============================] - 115s 8ms/step
acc: 0.8621333333651224 loss: 0.3276844515800476  highest: 0.8621333333651224
Epoch 2/3
15000/15000 [==============================] - 205s 14ms/step - loss: 0.2829 - acc: 0.8834
15000/15000 [==============================] - 99s 7ms/step
acc: 0.8818 loss: 0.2809721389929454  highest: 0.8818
Epoch 3/3
15000/15000 [==============================] - 202s 13ms/step - loss: 0.2268 - acc: 0.9070
15000/15000 [==============================] - 108s 7ms/step
acc: 0.9002 loss: 0.24286139675776164  highest: 0.9002
[0.8621333333651224, 0.8818, 0.9002]
Epoch 1/10
15000/15000 [==============================] - 187s 12ms/step - loss: 0.1201 - acc: 0.9480
15000/15000 [==============================] - 94s 6ms/step
acc: 0.8977333333333334 loss: 0.41479730946520965  highest: 0.9002
Epoch 2/10
15000/15000 [==============================] - 191s 13ms/step - loss: 0.0597 - acc: 0.9765
15000/15000 [==============================] - 89s 6ms/step
acc: 0.8999333333333334 loss: 0.49938425362209476  highest: 0.9002
Epoch 3/10
15000/15000 [==============================] - 189s 13ms/step - loss: 0.0472 - acc: 0.9822
15000/15000 [==============================] - 88s 6ms/step
acc: 0.8872666666666666 loss: 0.7023815796419978  highest: 0.9002
"""
