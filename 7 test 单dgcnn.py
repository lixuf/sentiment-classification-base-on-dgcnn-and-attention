



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

    x = DilatedGatedConv1D(rate=1, drop_gate=dro)([x, mask])
    x = DilatedGatedConv1D(rate=2, drop_gate=dro)([x, mask])
    x = DilatedGatedConv1D(rate=4, drop_gate=dro)([x, mask])
    x = DilatedGatedConv1D(rate=1, drop_gate=dro)([x, mask])
    x=AttentionPooling1D()([x,mask])
    return x

def block_cnn(x,mask,dro):
    x = Conv1D(175,kernel_size=3,padding='same', activation='relu')(x)
    x = Dropout(dro)(x)
    x = Dense(50,activation='relu')(x)
    x = Dropout(dro)(x)
    x= AttentionPooling1D()([x,mask])
    return x

def pv_con(x):
    x,pv=x
    return K.concatenate([x,pv],axis=-1)

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

def build_model_cnn(maxlen,word_size,data_vol,batch_size):
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
 

    x=Lambda(union_4)([x1,x2,x3,x4])
    p = Dense(1, activation='sigmoid')(x)
    model=Model([x_in,x_in_un],p)
    
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    return model

def build_model_dgcnn_pv(maxlen,word_size,data_vol,batch_size):
    pv_size=20
    x_in= Input(shape=(None,word_size))
    x_in_un = Input(shape=(None,))
    pid = Lambda(position_id)(x_in_un)
    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x_in_un)
    position_embedding = Embedding(maxlen, pv_size)
    pv = position_embedding(pid)
    x=Dense(word_size,use_bias=False)(x_in)
    pv=Lambda(mask_ed)([pv,mask])
    x =Lambda(pv_con,output_shape=(None,word_size+pv_size))([x, pv])


    x1=Lambda(spilt_4,output_shape=(None,word_size+pv_size),arguments={'index':1})(x)
    mask1=Lambda(spilt_4,output_shape=(None,),arguments={'index':1})(mask)

    x2=Lambda(spilt_4,output_shape=(None,word_size+pv_size),arguments={'index':2})(x)
    mask2=Lambda(spilt_4,output_shape=(None,),arguments={'index':2})(mask)

    x3=Lambda(spilt_4,output_shape=(None,word_size+pv_size),arguments={'index':3})(x)
    mask3=Lambda(spilt_4,output_shape=(None,),arguments={'index':3})(mask)

    x4=Lambda(spilt_4,output_shape=(None,word_size+pv_size),arguments={'index':4})(x)
    mask4=Lambda(spilt_4,output_shape=(None,),arguments={'index':4})(mask)

   

    x1 = block_dgcnn(x1,mask1,0.1)

    x2 = block_dgcnn(x2,mask2,0.2)

    x3 = block_dgcnn(x3,mask3,0.3)

    x4 = block_dgcnn(x4,mask4,0.5)
 

    x=Lambda(union_4)([x1,x2,x3,x4])
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
    history=model.fit([x_train,x_train_un],y_train,epochs=30,batch_size=batch_size,callbacks=[evaluator]) 
    print(evaluator.accs)
    
 """
 Relying on driver to perform ptx compilation. This message will be only logged once.
15000/15000 [==============================] - 184s 12ms/step - loss: 0.3794 - acc: 0.8254
15000/15000 [==============================] - 118s 8ms/step
acc: 0.8829333333651225 loss: 0.28517896067301435  highest: 0.8829333333651225
Epoch 2/30
15000/15000 [==============================] - 191s 13ms/step - loss: 0.2624 - acc: 0.8912
15000/15000 [==============================] - 119s 8ms/step
acc: 0.8806 loss: 0.2815084527810415  highest: 0.8829333333651225
Epoch 3/30
15000/15000 [==============================] - 209s 14ms/step - loss: 0.1782 - acc: 0.9287
15000/15000 [==============================] - 111s 7ms/step
acc: 0.8828666666348776 loss: 0.2931689668734868  highest: 0.8829333333651225
Epoch 4/30
15000/15000 [==============================] - 189s 13ms/step - loss: 0.1181 - acc: 0.9539
15000/15000 [==============================] - 120s 8ms/step
acc: 0.8799333333015442 loss: 0.3847924997528394  highest: 0.8829333333651225
Epoch 5/30
15000/15000 [==============================] - 210s 14ms/step - loss: 0.0936 - acc: 0.9645
15000/15000 [==============================] - 119s 8ms/step
acc: 0.8775333333651225 loss: 0.3923147148927053  highest: 0.8829333333651225
Epoch 6/30
15000/15000 [==============================] - 204s 14ms/step - loss: 0.0660 - acc: 0.9754
15000/15000 [==============================] - 113s 8ms/step
acc: 0.8674666666984558 loss: 0.46680302190581957  highest: 0.8829333333651225
Epoch 7/30
15000/15000 [==============================] - 204s 14ms/step - loss: 0.0719 - acc: 0.9726
15000/15000 [==============================] - 119s 8ms/step
acc: 0.8699333333333333 loss: 0.4927846217095852  highest: 0.8829333333651225
Epoch 8/30
15000/15000 [==============================] - 202s 13ms/step - loss: 0.0659 - acc: 0.9743
15000/15000 [==============================] - 118s 8ms/step
acc: 0.8764 loss: 0.4746832819541295  highest: 0.8829333333651225
Epoch 9/30
15000/15000 [==============================] - 208s 14ms/step - loss: 0.0686 - acc: 0.9755
15000/15000 [==============================] - 126s 8ms/step
acc: 0.8742666666348775 loss: 0.4878319390793641  highest: 0.8829333333651225
Epoch 10/30
15000/15000 [==============================] - 195s 13ms/step - loss: 0.0685 - acc: 0.9741
15000/15000 [==============================] - 123s 8ms/step
acc: 0.8652000000317891 loss: 0.5507168663620948  highest: 0.8829333333651225
Epoch 11/30
15000/15000 [==============================] - 207s 14ms/step - loss: 0.0445 - acc: 0.9840
15000/15000 [==============================] - 122s 8ms/step
acc: 0.8550666666984558 loss: 0.7245310015122096  highest: 0.8829333333651225
Epoch 12/30
15000/15000 [==============================] - 187s 12ms/step - loss: 0.0540 - acc: 0.9810
15000/15000 [==============================] - 129s 9ms/step
acc: 0.8761999999682109 loss: 0.5212171302060287  highest: 0.8829333333651225
Epoch 13/30
15000/15000 [==============================] - 202s 13ms/step - loss: 0.0495 - acc: 0.9823
15000/15000 [==============================] - 122s 8ms/step
acc: 0.8722 loss: 0.531897952268521  highest: 0.8829333333651225
Epoch 14/30
15000/15000 [==============================] - 199s 13ms/step - loss: 0.0425 - acc: 0.9842
15000/15000 [==============================] - 123s 8ms/step
acc: 0.8813333333015442 loss: 0.5570229072729747  highest: 0.8829333333651225
Epoch 15/30
15000/15000 [==============================] - 173s 12ms/step - loss: 0.0519 - acc: 0.9809
15000/15000 [==============================] - 143s 10ms/step
acc: 0.8600666666984558 loss: 0.6598413528223832  highest: 0.8829333333651225
Epoch 16/30
15000/15000 [==============================] - 199s 13ms/step - loss: 0.0763 - acc: 0.9737
15000/15000 [==============================] - 118s 8ms/step
acc: 0.8594666666666667 loss: 0.5815899517695109  highest: 0.8829333333651225
Epoch 17/30
15000/15000 [==============================] - 209s 14ms/step - loss: 0.0667 - acc: 0.9762
15000/15000 [==============================] - 117s 8ms/step
acc: 0.8763333333015442 loss: 0.551939327599605  highest: 0.8829333333651225
Epoch 18/30
15000/15000 [==============================] - 191s 13ms/step - loss: 0.0756 - acc: 0.9729
15000/15000 [==============================] - 124s 8ms/step
acc: 0.8671333333333333 loss: 0.541233625570933  highest: 0.8829333333651225
Epoch 19/30
15000/15000 [==============================] - 202s 13ms/step - loss: 0.0667 - acc: 0.9755
15000/15000 [==============================] - 122s 8ms/step
acc: 0.8586666666348776 loss: 0.6067776434520881  highest: 0.8829333333651225
Epoch 20/30
15000/15000 [==============================] - 205s 14ms/step - loss: 0.0596 - acc: 0.9788
15000/15000 [==============================] - 117s 8ms/step
acc: 0.8508666666984558 loss: 0.7084074107209841  highest: 0.8829333333651225
Epoch 21/30
15000/15000 [==============================] - 197s 13ms/step - loss: 0.0852 - acc: 0.9698
15000/15000 [==============================] - 123s 8ms/step
acc: 0.8706 loss: 0.5010321761687596  highest: 0.8829333333651225
Epoch 22/30
15000/15000 [==============================] - 178s 12ms/step - loss: 0.0887 - acc: 0.9689
15000/15000 [==============================] - 95s 6ms/step
acc: 0.8618000000317891 loss: 0.628796885784467  highest: 0.8829333333651225
Epoch 23/30
15000/15000 [==============================] - 168s 11ms/step - loss: 0.0998 - acc: 0.9639
15000/15000 [==============================] - 94s 6ms/step
acc: 0.8735999999682108 loss: 0.5063600245952606  highest: 0.8829333333651225
Epoch 24/30
15000/15000 [==============================] - 155s 10ms/step - loss: 0.1386 - acc: 0.9503
15000/15000 [==============================] - 93s 6ms/step
acc: 0.8526666666348776 loss: 0.5823375865856807  highest: 0.8829333333651225
Epoch 25/30
15000/15000 [==============================] - 171s 11ms/step - loss: 0.1788 - acc: 0.9368
15000/15000 [==============================] - 90s 6ms/step
acc: 0.8496666666666667 loss: 0.5430574720859528  highest: 0.8829333333651225
Epoch 26/30
15000/15000 [==============================] - 170s 11ms/step - loss: 0.2196 - acc: 0.9217
15000/15000 [==============================] - 94s 6ms/step
acc: 0.8634000000317892 loss: 0.4626811152935028  highest: 0.8829333333651225
Epoch 27/30
15000/15000 [==============================] - 170s 11ms/step - loss: 0.2319 - acc: 0.9150
15000/15000 [==============================] - 95s 6ms/step
acc: 0.8643999999682108 loss: 0.446978994790713  highest: 0.8829333333651225
Epoch 28/30
15000/15000 [==============================] - 171s 11ms/step - loss: 0.2370 - acc: 0.9181
15000/15000 [==============================] - 99s 7ms/step
acc: 0.8623333333651225 loss: 0.41600582197507224  highest: 0.8829333333651225
Epoch 29/30
15000/15000 [==============================] - 179s 12ms/step - loss: 0.3038 - acc: 0.9023
15000/15000 [==============================] - 102s 7ms/step
acc: 0.8562 loss: 0.43566155507564547  highest: 0.8829333333651225
Epoch 30/30
15000/15000 [==============================] - 171s 11ms/step - loss: 0.4716 - acc: 0.8689
15000/15000 [==============================] - 96s 6ms/step
acc: 0.8231333333015441 loss: 0.7174816767295201  highest: 0.8829333333651225
[0.8829333333651225, 0.8806, 0.8828666666348776, 0.8799333333015442, 0.8775333333651225, 0.8674666666984558, 0.8699333333333333, 0.8764, 0.8742666666348775, 0.8652000000317891, 0.8550666666984558, 0.8761999999682109, 0.8722, 0.8813333333015442, 0.8600666666984558, 0.8594666666666667, 0.8763333333015442, 0.8671333333333333, 0.8586666666348776, 0.8508666666984558, 0.8706, 0.8618000000317891, 0.8735999999682108, 0.8526666666348776, 0.8496666666666667, 0.8634000000317892, 0.8643999999682108, 0.8623333333651225, 0.8562, 0.8231333333015441]
"""