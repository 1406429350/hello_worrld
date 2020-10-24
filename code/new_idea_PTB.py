#-*coding: UTF-8 -*-
import warnings
warnings.filterwarnings('ignore')
import keras
import h5py
import numpy as np
np.random.seed(1337)
from keras.layers import Input, Dense, Convolution1D, MaxPool1D, Activation, RepeatVector, Dropout, BatchNormalization, SpatialDropout1D, Flatten, Convolution2D, MaxPooling2D, Reshape
from keras.models import load_model
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
import scipy.io as sio
import random
import matplotlib.pyplot as plt
from keras import regularizers
import tensorflow as tf
from keras.utils import plot_model
from IPython.display import Image

n_class = 2
epochs = 10000    # 训练次数

batch_size = 128   # 每一类的批数

length = 800     # 截取的数据
half_length = 400    # 截取一段数据的一半

n_hidden_units = 100   # neurons in hidden layer
# -------------------------------------数据准备-----------------------------------------------------------------------------------

train_path = 'train_ptb.mat'
test_path = 'test_ptb.mat'
train_data = sio.loadmat(train_path)
test_path = sio.loadmat(test_path)
y_train = train_data["train_ptb"]
y_test = test_path["test_ptb"]
# file = h5py.File('model.h5','w')

# --------------------训练集---------------------------------------

m, n = y_train.shape
Y_train = y_train[-1, :]  # 获取标签
print(Y_train[15:25])
Y = np.transpose(y_train)  # 转置
plt.figure(1)

X_train_lead1 = Y[:, 0:half_length]  # 获取一个导联的一条前数据
plt.subplot(3, 4, 1)
plt.plot(X_train_lead1[50, :])

X_train_lead1 = X_train_lead1.reshape([-1, half_length, 1])
X_train_lead2 = Y[:, half_length:length]
X_train_lead2 = X_train_lead2.reshape([-1, half_length, 1])
X_train_lead3 = Y[:, length:3*half_length]
X_train_lead3 = X_train_lead3.reshape([-1, half_length,1])
X_train_lead4 = Y[:, 3*half_length:2*length]
X_train_lead4 = X_train_lead4.reshape([-1, half_length,1])

X_train_lead5 = Y[:, 2*length:5*half_length]
X_train_lead5 = X_train_lead5.reshape([-1, half_length, 1])
X_train_lead6 = Y[:, 5*half_length:3*length]
X_train_lead6 = X_train_lead6.reshape([-1, half_length, 1])
X_train_lead7 = Y[:, 3*length:7*half_length]
X_train_lead7 = X_train_lead7.reshape([-1, half_length, 1])
X_train_lead8 = Y[:, 7*half_length:4*length]
X_train_lead8 = X_train_lead8.reshape([-1, half_length, 1])
X_train_lead9 = Y[:, 4*length:half_length*9]
X_train_lead9 = X_train_lead9.reshape([-1, half_length, 1])
X_train_lead10 = Y[:, 9*half_length:5*length]
X_train_lead10 = X_train_lead10.reshape([-1, half_length, 1])
X_train_lead11 = Y[:, 5*length:11*half_length]
X_train_lead11 = X_train_lead11.reshape([-1, half_length, 1])
X_train_lead12 = Y[:, 11*half_length:6*length]
X_train_lead12 = X_train_lead12.reshape([-1, half_length, 1])

plt.subplot(3, 4, 2)
plt.plot(X_train_lead2[50, :])
plt.subplot(3, 4, 3)
plt.plot(X_train_lead3[50, :])
plt.subplot(3, 4, 4)
plt.plot(X_train_lead4[50, :])

plt.subplot(3, 4, 5)
plt.plot(X_train_lead5[50, :])
plt.subplot(3, 4, 6)
plt.plot(X_train_lead6[50, :])
plt.subplot(3, 4, 7)
plt.plot(X_train_lead7[50, :])
plt.subplot(3, 4, 8)
plt.plot(X_train_lead8[50, :])
plt.subplot(3, 4, 9)
plt.plot(X_train_lead9[50, :])
plt.subplot(3, 4, 10)
plt.plot(X_train_lead10[50, :])
plt.subplot(3, 4, 11)
plt.plot(X_train_lead11[50, :])
plt.subplot(3, 4, 12)
plt.plot(X_train_lead12[50, :])
plt.show()

Y_train=np_utils.to_categorical(Y_train, num_classes=n_class)
print('X_train_lead1:', X_train_lead1.shape)
print('X_train_lead2:', X_train_lead2.shape)
print('X_train_lead3:', X_train_lead3.shape)
print('X_train_lead4:', X_train_lead4.shape)
print('Y_train:', Y_train.shape)
print("Label:", Y_train[15:25, :])

# --------------------测试集----------------------------------------

m1, n1 = y_test.shape
Y_test = y_test[-1, :]
label = Y_test
print(label[15:25])
Y1 = np.transpose(y_test)

X_test_lead1 = Y1[:, 0:half_length]

plt.subplot(3, 4, 1)
plt.plot(X_test_lead1[50, :])

X_test_lead1 = X_test_lead1.reshape([-1, half_length,1])
X_test_lead2 = Y1[:, half_length:length]
X_test_lead2 = X_test_lead2.reshape([-1, half_length,1])
X_test_lead3 = Y1[:, length:half_length*3]
X_test_lead3 = X_test_lead3.reshape([-1, half_length, 1])
X_test_lead4 = Y1[:, 3*half_length:2*length]
X_test_lead4 = X_test_lead4.reshape([-1, half_length,1])

X_test_lead5 = Y1[:, 2*length:half_length*5]
X_test_lead5 = X_test_lead5.reshape([-1, half_length, 1])
X_test_lead6 = Y1[:, half_length*5:3*length]
X_test_lead6 = X_test_lead6.reshape([-1, half_length, 1])
X_test_lead7 = Y1[:, 3*length:7*half_length]
X_test_lead7 = X_test_lead7.reshape([-1, half_length, 1])
X_test_lead8 = Y1[:, 7*half_length:4*length]
X_test_lead8 = X_test_lead8.reshape([-1, half_length, 1])
X_test_lead9 = Y1[:, 4*length:half_length*9]
X_test_lead9 = X_test_lead9.reshape([-1, half_length, 1])
X_test_lead10 = Y1[:, 9*half_length:5*length]
X_test_lead10 = X_test_lead10.reshape([-1, half_length, 1])
X_test_lead11 = Y1[:, 5*length:half_length*11]
X_test_lead11 = X_test_lead11.reshape([-1, half_length, 1])
X_test_lead12 = Y1[:, 11*half_length:6*length]
X_test_lead12 = X_test_lead12.reshape([-1, half_length, 1])

plt.subplot(3, 4, 2)
plt.plot(X_test_lead2[50, :])
plt.subplot(3, 4, 3)
plt.plot(X_test_lead3[50, :])
plt.subplot(3, 4, 4)
plt.plot(X_test_lead4[50, :])

plt.subplot(3, 4, 5)
plt.plot(X_test_lead5[50, :])
plt.subplot(3, 4, 6)
plt.plot(X_test_lead6[50, :])
plt.subplot(3, 4, 7)
plt.plot(X_test_lead7[50, :])
plt.subplot(3, 4, 8)
plt.plot(X_test_lead8[50, :])
plt.subplot(3, 4, 9)
plt.plot(X_test_lead9[50,:])
plt.subplot(3,4,10)
plt.plot(X_test_lead10[50, :])
plt.subplot(3, 4, 11)
plt.plot(X_test_lead11[50, :])
plt.subplot(3, 4, 12)
plt.plot(X_test_lead12[50, :])
plt.show()

Y_test = np_utils.to_categorical(Y_test, num_classes=n_class)
print('X_test_lead1:', X_test_lead1.shape)
print('X_test_lead2:', X_test_lead2.shape)
print('X_test_lead3:', X_test_lead3.shape)
print('X_test_lead4:', X_test_lead4.shape)
print('Y_test:', Y_test.shape)
print("Label:", Y_test[15:25,:])
# -----------------------------------模型-------------------------------------------------------------------------------------------

lead1 = Input(shape=(half_length,1))
lead2 = Input(shape=(half_length,1))
lead3 = Input(shape=(half_length,1))
lead4 = Input(shape=(half_length,1))
lead5 = Input(shape=(half_length,1))
lead6 = Input(shape=(half_length,1))
lead7 = Input(shape=(half_length,1))
lead8 = Input(shape=(half_length,1))
lead9 = Input(shape=(half_length,1))
lead10 = Input(shape=(half_length,1))
lead11 = Input(shape=(half_length,1))
lead12 = Input(shape=(half_length,1))



# 定义网络层，3层卷积，3层池化，3层全连接
# 1
merge_CNN11 = Convolution1D(5, 3, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(lead1)

merge_pool11 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN11)

merge_CNN21 = Convolution1D(10, 4, strides=1, padding='same', activation='relu')(merge_pool11)

merge_pool21 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN21)

merge_CNN31 = Convolution1D(20, 4, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(merge_pool21)

merge_pool31 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN31)

# 2
merge_CNN12 = Convolution1D(5, 3, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(lead2)

merge_pool12 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN12)

merge_CNN22 = Convolution1D(10, 4, strides=1, padding='same', activation='relu')(merge_pool12)

merge_pool22 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN22)

merge_CNN32 = Convolution1D(20, 4, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(merge_pool22)

merge_pool32 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN32)

# 3
merge_CNN13 = Convolution1D(5, 3, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(lead3)

merge_pool13 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN13)

merge_CNN23 = Convolution1D(10, 4, strides=1, padding='same', activation='relu')(merge_pool13)

merge_pool23 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN23)

merge_CNN33 = Convolution1D(20, 4, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(merge_pool23)

merge_pool33 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN33)

# 4
merge_CNN14 = Convolution1D(5, 3, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(lead4)

merge_pool14 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN14)

merge_CNN24 = Convolution1D(10, 4, strides=1, padding='same', activation='relu')(merge_pool14)

merge_pool24 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN24)

merge_CNN34 = Convolution1D(20, 4, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(merge_pool24)

merge_pool34 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN34)

# 5

merge_CNN15 = Convolution1D(5, 3, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(lead5)

merge_pool15 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN15)

merge_CNN25 = Convolution1D(10, 4, strides=1, padding='same', activation='relu')(merge_pool15)

merge_pool25 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN25)

merge_CNN35 = Convolution1D(20, 4, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(merge_pool25)

merge_pool35 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN35)

# 6
merge_CNN16 = Convolution1D(5, 3, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(lead6)

merge_pool16 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN16)

merge_CNN26 = Convolution1D(10, 4, strides=1, padding='same', activation='relu')(merge_pool16)

merge_pool26 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN26)

merge_CNN36 = Convolution1D(20, 4, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(merge_pool26)

merge_pool36 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN36)

# 7
merge_CNN17 = Convolution1D(5, 3, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(lead7)

merge_pool17 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN17)

merge_CNN27 = Convolution1D(10, 4, strides=1, padding='same', activation='relu')(merge_pool17)

merge_pool27 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN27)

merge_CNN37 = Convolution1D(20, 4, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(merge_pool27)

merge_pool37 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN37)

# 8
merge_CNN18 = Convolution1D(5, 3, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(lead8)

merge_pool18 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN18)

merge_CNN28 = Convolution1D(10, 4, strides=1, padding='same', activation='relu')(merge_pool18)

merge_pool28 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN28)

merge_CNN38 = Convolution1D(20, 4, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(merge_pool28)

merge_pool38 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN38)

# 9
merge_CNN19 = Convolution1D(5, 3, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(lead9)

merge_pool19 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN19)

merge_CNN29 = Convolution1D(10, 4, strides=1, padding='same', activation='relu')(merge_pool19)

merge_pool29 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN29)

merge_CNN39 = Convolution1D(20, 4, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(merge_pool29)

merge_pool39 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN39)

# 10
merge_CNN110 = Convolution1D(5, 3, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(lead10)

merge_pool110 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN110)

merge_CNN210 = Convolution1D(10, 4, strides=1, padding='same', activation='relu')(merge_pool110)

merge_pool210 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN210)

merge_CNN310 = Convolution1D(20, 4, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(merge_pool210)

merge_pool310 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN310)

# 11
merge_CNN111 = Convolution1D(5, 3, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(lead11)

merge_pool111 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN111)

merge_CNN211 = Convolution1D(10, 4, strides=1, padding='same', activation='relu')(merge_pool111)

merge_pool211 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN211)

merge_CNN311 = Convolution1D(20, 4, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(merge_pool211)

merge_pool311 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN311)

# 12
merge_CNN112 = Convolution1D(5, 3, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(lead12)

merge_pool112 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN112)

merge_CNN212 = Convolution1D(10, 4, strides=1, padding='same', activation='relu')(merge_pool112)

merge_pool212 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN212)

merge_CNN312 = Convolution1D(20, 4, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(merge_pool212)

merge_pool312 = MaxPool1D(pool_size=2, strides=2, padding='same')(merge_CNN312)

# merge_CNN2D = Convolution2D(filters=16, kernel_size=(3, 3), strides=(1, 1),
#                             input_shape=(50, 20, 1), activation='relu')

# merge_pool2D = MaxPooling2D(pool_size=2, strides=2, padding='same')

# 合并多导连，网络层的输入
Lead = keras.layers.concatenate([merge_pool31, merge_pool32, merge_pool33, merge_pool34, merge_pool35, merge_pool36, merge_pool37,
                                 merge_pool38, merge_pool39, merge_pool310, merge_pool311, merge_pool312], axis=-1)  # 组合连接

CNN_all1 = Convolution1D(5, 3, strides=1, padding='same', input_shape=(lead1[1], lead1[2]), activation='relu')(Lead)

pool_all1 = MaxPool1D(pool_size=2, strides=2, padding='same')(CNN_all1)

CNN_all2 = Convolution1D(10, 4, strides=1, padding='same', activation='relu')(pool_all1)

pool_all2 = MaxPool1D(pool_size=2, strides=2, padding='same')(CNN_all2)
print('merge_lead:', Lead.shape)


Lead = Flatten()(pool_all2)  # 把卷积所得到的特征平铺开
print('Lead10:', Lead.shape)
# Lead = Dense(128, activation='relu')(Lead)
Lead = Dense(64, activation='relu')(Lead)
# Lead = Dense(32, activation='relu')(Lead)
Lead = Dense(16, activation='relu')(Lead)
pred = Dense(n_class, activation='softmax')(Lead)
# print(pred.shape)

model = Model(inputs=[lead1, lead2, lead3, lead4, lead5, lead6, lead7, lead8, lead9, lead10, lead11, lead12], outputs=pred)


model.summary()   # 模型打印


# 产生网络拓扑图
#plot_model(model, to_file='recurrent_neural_network.png')


# Image('recurrent_neural_network.png')

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# ----------------------------------------训练------------------------------------------------------------------------------------------------

batch_index = 0

for step in range(epochs):
    X_batch1 = X_train_lead1[batch_index:batch_index+batch_size, :, :]
    X_batch2 = X_train_lead2[batch_index:batch_index+batch_size, :, :]
    X_batch3 = X_train_lead3[batch_index:batch_index+batch_size, :, :]
    X_batch4 = X_train_lead4[batch_index:batch_index+batch_size, :, :]
    X_batch5 = X_train_lead5[batch_index:batch_index+batch_size, :, :]
    X_batch6 = X_train_lead6[batch_index:batch_index+batch_size, :, :]
    X_batch7 = X_train_lead7[batch_index:batch_index+batch_size, :, :]
    X_batch8 = X_train_lead8[batch_index:batch_index+batch_size, :, :]
    X_batch9 = X_train_lead9[batch_index:batch_index+batch_size, :, :]
    X_batch10 = X_train_lead10[batch_index:batch_index+batch_size, :, :]
    X_batch11 = X_train_lead11[batch_index:batch_index+batch_size, :, :]
    X_batch12 = X_train_lead12[batch_index:batch_index+batch_size, :, :]
    Y_batch = Y_train[batch_index:batch_index+batch_size, :]
    cost = model.train_on_batch([X_batch1, X_batch2, X_batch3, X_batch4, X_batch5, X_batch6, X_batch7, X_batch8, X_batch9, X_batch10, X_batch11, X_batch12], Y_batch)
    batch_index += batch_size
    batch_index = 0 if batch_index >= X_train_lead1.shape[0] else batch_index

    if step % 50 == 0:
        cost, acc = model.evaluate([X_test_lead1, X_test_lead2, X_test_lead3, X_test_lead4, X_test_lead5,
                                    X_test_lead6, X_test_lead7, X_test_lead8, X_test_lead9, X_test_lead10, X_test_lead11, X_test_lead12], Y_test, verbose=False)   #X_test_lead1
        print("Steps", step, ":")
        print('cost:', cost, 'acc:', acc)
        if acc > 0.99:
            result = model.predict([X_test_lead1, X_test_lead2, X_test_lead3, X_test_lead4,
                                    X_test_lead5, X_test_lead6, X_test_lead7, X_test_lead8, X_test_lead9, X_test_lead10, X_test_lead11, X_test_lead12])
            pred = np.argmax(result, axis=1)

            save_name = 'test_in'+str(step)+'.mat'
            sio.savemat(save_name, {'pred': pred, 'label': label, 'Acc': acc})
            print("This is step", step)



# model.save_weights('my_model_weights.h5')
# model.save('model.h5')
# file.close()
# del model
# model.save_weights(filepath)
# model.save('model3.h5')

# K.clear_session()
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#      json_file.write(model_json)

# model.save_weights("model.h5")

# print('saved')

