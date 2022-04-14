import numpy as np

x_train1=np.load('../dataset/data/x_train1.npy')
x_train2=np.load('../dataset/data/x_train2.npy')
x_train_w1=np.load('../dataset/data/x_train_w1.npy')
x_train_w2=np.load('../dataset/data/x_train_w2.npy')
y_train=np.load('../dataset/data/y_train.npy')
x_val1=np.load('../dataset/data/x_val1.npy')
x_val2=np.load('../dataset/data/x_val2.npy')
x_val_w1=np.load('../dataset/data/x_val_w1.npy')
x_val_w2=np.load('../dataset/data/x_val_w2.npy')
y_val=np.load('../dataset/data/y_val.npy')
x_test1=np.load('../dataset/data/x_test1.npy')
x_test2=np.load('../dataset/data/x_test2.npy')
x_test_w1=np.load('../dataset/data/x_test_w1.npy')
x_test_w2=np.load('../dataset/data/x_test_w2.npy')
y_test=np.load('../dataset/data/y_test.npy')

x_pos_train1= np.load('../dataset/data/x_pos_train1.npy') / 34
x_pos_train2= np.load('../dataset/data/x_pos_train2.npy') / 34
x_pos_val1= np.load('../dataset/data/x_pos_val1.npy') / 34
x_pos_val2= np.load('../dataset/data/x_pos_val2.npy') / 34
x_pos_test1= np.load('../dataset/data/x_pos_test1.npy') / 34
x_pos_test2= np.load('../dataset/data/x_pos_test2.npy') / 34


embedding_matrix=np.load('../dataset/data/embedding_matrix_100_mincount_1.npy')
x_char_train1=np.load('../dataset/char/x_train1.npy')
x_char_train2=np.load('../dataset/char/x_train2.npy')

x_char_val1=np.load('../dataset/char/x_val1.npy')
x_char_val2=np.load('../dataset/char/x_val2.npy')

x_char_test1=np.load('../dataset/char/x_test1.npy')
x_char_test2=np.load('../dataset/char/x_test2.npy')

# np.random.seed(5689)
maxlen = 15  # We will cut reviews after 30 words
max_words = 5257  # We will only consider the top 10,000 words in the dataset
embedding_dim = 100
char_len=5
max_chars=1625
char_dim=50
char_embdim=50
batchsize=32

from keras.models import Model
from keras.layers import Dense, Embedding, Input, Lambda, BatchNormalization
from keras.layers import LSTM, Bidirectional,Conv1D,Flatten
import keras.backend as K
from keras.layers import Concatenate
from keras.layers.merge import concatenate
import tensorflow as tf

np.random.seed(5689)
tf.set_random_seed(5689)


print('build model...')
input1 = Input(shape=(maxlen,), dtype='float32') #预期的输入是15维向量 15是句子的最大长度
input2 = Input(shape=(maxlen,), dtype='float32')
input3 = Input(shape=(maxlen,), dtype='float32')
input4 = Input(shape=(maxlen,), dtype='float32')
input5 = Input(shape=(maxlen,), dtype='float32')
input6 = Input(shape=(maxlen,), dtype='float32')
input7 = Input(shape=(maxlen,5), dtype='float32')
input8 = Input(shape=(maxlen,5), dtype='float32')

embedder2 = Embedding(max_words, embedding_dim, input_length=maxlen,mask_zero=True,weights = [embedding_matrix], trainable = False)

charEmbedder=Embedding(max_chars, char_dim, input_length=(maxlen,char_len),mask_zero=True, trainable = True)

def same_word1(x):
    x=K.reshape(x,(-1,maxlen,1))
    return x

embedder3=Lambda(same_word1,output_shape=(maxlen,1))(input3)
embedder4=Lambda(same_word1,output_shape=(maxlen,1))(input4)
embedder5=Lambda(same_word1,output_shape=(maxlen,1))(input5)
embedder6=Lambda(same_word1,output_shape=(maxlen,1))(input6)

ci7=charEmbedder(input7)
# ss=K.sum(ci7,axis=2)
#(batch,25,5,100)
ci8=charEmbedder(input8)


def rsp(x):
    ss=K.reshape(x,(-1,x.shape[2],x.shape[3]))
    # sc=char_gru(ss)
    return ss

c7=Lambda(rsp,output_shape=(char_len,char_dim))(ci7)
c8=Lambda(rsp,output_shape=(char_len,char_dim))(ci8)

char_gru=LSTM(char_embdim, dropout=0.2, recurrent_dropout=0.1)
cr7=char_gru(c7)
cr8=char_gru(c8)
# cc7=Lambda(exp_dim,output_shape=(maxlen,x.shape))(ci7)

def char_emb(x,dim=char_embdim):
    ss=K.reshape(x,(-1,maxlen,dim))
    return ss
# embedder7=Lambda(K.max,arguements={'axis':2})(ci7)
# embedder8=Lambda(K.max,arguements={'axis':2})(ci8)

embedder7=Lambda(char_emb,output_shape=(maxlen,char_embdim))(cr7)
embedder8=Lambda(char_emb,output_shape=(maxlen,char_embdim))(cr8)
# embedder7=Lambda(K.max, arguments={'axis' : 2})(ci7)
# embedder8=Lambda(K.max, arguments={'axis' : 2})(ci8)
embed1=concatenate([embedder2(input1),embedder3,embedder5,embedder7], axis=-1)
embed2=concatenate([embedder2(input2),embedder4,embedder6,embedder8], axis=-1)

# embed1=concatenate([embedder1(input1),embedder2(input1),embedder3,embedder5,embedder7], axis=-1)
# embed2=concatenate([embedder1(input2),embedder2(input2),embedder4,embedder6,embedder8], axis=-1)

# from Dlayer import Attention


from mylayers.DYlayer import WKS as Attention
from mylayers.DYlayer import CrossAttention
from mylayers.seq_self_attention import SeqSelfAttention
from mylayers.directional_self_attention import SeqDiSelfAttention
from mylayers.distance_self_attention1 import SeqDistanceSelfAttention
from mylayers.MatchLayer import MatchLayer

share_bLSTM1 = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.1,return_sequences=True))

l1 = share_bLSTM1(embed1)
r1 = share_bLSTM1(embed2)

share_bLSTM2 = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.1,return_sequences=True))

l2=share_bLSTM2(l1)
r2=share_bLSTM2(r1)

ll2= SeqDistanceSelfAttention(attention_activation='sigmoid',attention_distance="mydistance")(l2)
rr2= SeqDistanceSelfAttention(attention_activation='sigmoid',attention_distance="mydistance")(r2)

# al2,ar2,wp_l2,wp_r2,word_l2,word_r2=Attention(unit=16)([l2,r2])
wp_l2,wp_r2=CrossAttention(unit=15)([l2,r2])

ct_l=concatenate([ll2,wp_l2],axis=-1)
ct_r=concatenate([rr2,wp_r2],axis=-1)


#多level match
match1,match2,match3 = MatchLayer()([ct_l,ct_r])

#聚合 Aggregation
final_match = concatenate([match1,match2,match3],axis=-1)
# cnn1 = Conv1D(300, 3, padding='same', strides=1, activation='relu')
# q_conv1 = cnn1(final_match)
# q_flat = Flatten()(q_conv1)

mx1 = Lambda(K.max, arguments={'axis' : 1})(final_match) #(None,  2310)
av1 = Lambda(K.mean, arguments={'axis' : 1})(final_match)

y = Concatenate()([av1, mx1])



# # y = Dropout(0.5)(y)
y = BatchNormalization()(y)
y = Dense(64, activation='relu')(y)


merged = BatchNormalization()(y)
merged = Dense(16, activation='relu')(merged)

# merged = Dropout(0.5)(merged)
merged = BatchNormalization()(merged)
output = Dense(1, activation='sigmoid')(merged)


model = Model(inputs = [input1,input2,input3,input4,input5,input6,input7,input8], outputs = output)
model.summary()



from keras.callbacks import EarlyStopping,ModelCheckpoint
import os
path=r'../savemodel/dy'
if not os.path.exists(path):
    os.mkdir(path)


saveBestModel = ModelCheckpoint(filepath = "../savemodel/dy/dy_{epoch:03d}-{acc:.4f}.h5", monitor='acc', verbose=2, mode='auto')

from keras.callbacks import Callback


def get_median(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0:   # 判断列表长度为偶数
        median = (data[size//2]+data[size//2-1])/2
        data[0] = median
    if size % 2 == 1:   # 判断列表长度为奇数
        median = data[(size-1)//2]
        data[0] = median
    return data[0]

# def evaluateAcc(data,label)

class  TestAcc(Callback):
    def __init__(self,data,label,msg):
        self.data=data
        self.label=label
        self.msg=msg
    def on_epoch_end(self,epoch, logs=None):
        pre=self.model.predict(self.data)
        loss,acc_eval=self.model.evaluate(self.data,self.label)
        th=get_median(pre)
        s1=[1 if s>th else 0 for s in pre]
        label1=self.label
        acc=1-sum(abs(s1-label1))/len(s1)
        f=open(r'../output/distance mask/final_test.txt', 'a', encoding='utf-8')
        f.write(self.msg+'th='+str(th)+'myacc= '+str(acc)+'eval_acc='+' '+str(acc_eval)+'\n')
        f.close()
        print(self.msg+' msg:myacc='+' '+str(acc)+'eval_acc='+' '+str(acc_eval))


showtestacc=TestAcc(data=[x_test1,x_test2,x_test_w1,x_test_w2,x_pos_test1,x_pos_test2,x_char_test1,x_char_test2],label=y_test,msg='Test')

showvalacc=TestAcc(data=[x_val1,x_val2,x_val_w1,x_val_w2,x_pos_val1,x_pos_val1,x_char_val1,x_char_val2],label=y_val,msg='Val')

from keras.optimizers import RMSprop
opt=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
#--------------mew loss function
margin = 0.7
theta = lambda t: (K.sign(t)+1.)/2.
nb_classes = 2
def new_mse_loss(y_true, y_pred):
    loss1 = mse_loss(y_true, y_pred)
    #one_hot
    loss2 = mse_loss(K.ones_like(y_pred)/nb_classes, y_pred)
    return 0.9*loss1+0.1*loss2

def mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

def loss(y_true, y_pred):
    return - (1 - theta(y_true - margin) * theta(y_pred - margin)
              - theta(1 - margin - y_true) * theta(1 - margin - y_pred)
              ) * (y_true * K.log(y_pred + 1e-8) + (1 - y_true) * K.log(1 - y_pred + 1e-8))
def myloss(y_true, y_pred, e=0.35):
    loss1 = mse_loss(y_true, y_pred)
    #one_hot
    loss2 = mse_loss(K.ones_like(y_pred)/nb_classes, y_pred)
    loss3 = loss(y_true, y_pred)
    return e*loss1 + (1-e)*loss3
#loss=[focal_loss(alpha=.25, gamma=2)]
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit([x_train1,x_train2,x_train_w1,x_train_w2,x_pos_train1,x_pos_train2,x_char_train1,x_char_train2],y_train,
                    epochs=20,
                    batch_size=batchsize,
                    validation_data=([x_val1,x_val2,x_val_w1,x_val_w2,x_pos_val1,x_pos_val1,x_char_val1,x_char_val2],y_val),
                    callbacks=[saveBestModel,showvalacc,showtestacc])