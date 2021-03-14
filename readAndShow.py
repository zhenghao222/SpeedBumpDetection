import codecs
import numpy as np
import keras
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import random
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import sklearn
import os
from utils.input_data import read_data_sets
import utils.datasets as ds
import utils.augmentation as aug
import utils.helper as hlp

from keras import backend as K
from attention import Attention
from sklearn.model_selection import train_test_split
#from classifiers.encoder import Classifier_ENCODER

import time


f = codecs.open('./201023174101.txt',mode='r',encoding='utf-8')
#前两行是表头，跳过读取
line = f.readline()
line = f.readline()
#第一行数据
line = f.readline()
listdata = []
while line:
    a = line.split('\t')
    #垂直放置时，x方向为振动信号，取第3列
    col = a[2:3]
    listdata.append(col)
    line = f.readline()
#将string类型的list转为float类型的list
float_list = []
for i in listdata:
   float_list.append(float(i[0]))
print(float_list)
#通过matplotlib绘图
#创建绘图空间，1代表横坐标开始坐标，第二个参数代表结束坐标，第三个为长度

list_draw = float_list[0:2000]
numpy_raw = np.array(list_draw)
numpy_raw = numpy_raw.ravel()
hlp.plot1d(numpy_raw)
numpy_wrap = np.array(list_draw[100:500])
numpy_wrap = numpy_wrap.reshape((1,numpy_wrap.shape[0],1))
print(numpy_wrap.shape)
xval = np.arange(199)

#print(xval.shape)
y = numpy_wrap[0]
y = y.ravel()
x = np.arange(0,200,0.5).astype(int)
yval = np.interp(xval,x,y)
print(yval.shape)
'''
res_draw = np.concatenate((numpy_raw[0:100],yval),axis=0)
res_draw = np.concatenate((res_draw,numpy_raw[500:800]),axis=0)
hlp.plot1d(res_draw)
hlp.plot1d(yval)
hlp.plot1d(y)
'''
#plt.plot(xval,yval)

#numpy_wraped = aug.window_warp(numpy_wrap_temp,window_ratio=0.1,scales=[2,2])[0]
#hlp.plot1d(numpy_wraped)
#hlp.plot1d(numpy_wrap_temp[0])
'''

'''
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
x = np.linspace(1,len(list_draw),len(list_draw))
plt.plot(x,list_draw)
#使用字典rcParams访问并修改已加载的配置项
#使图形中的中文正常编码显示
plt.rcParams['font.sans-serif']=['SimHei']
#使坐标轴刻度表签正常显示正负号(False可正常显示)
plt.rcParams['axes.unicode_minus'] = False
plt.title('垂向振动',fontsize=30)
plt.xlabel('points',fontsize = 20)
plt.ylabel('ax(g)',fontsize = 20)
#显示图表
plt.show()



'''
#制作一维CNN数据集
def write_csv(Data,csv_file_name = 'train.csv'):
    with open(csv_file_name,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        beg_pos = 0
        while beg_pos+500 <= len(Data):
            list_fivehundred = Data[beg_pos:beg_pos+500]
            writer.writerow(list_fivehundred)
            beg_pos+=500
    csvfile.close()

write_csv(float_list)


#读取csv并画图展示
with open('train.csv','r') as f:
    reader = csv.reader(f)
    result_row = list(reader)
    #方法一：list转numpy（推荐：方便后续深度学习）
    char_numpy = np.array(result_row)
    float_numpy = np.zeros(char_numpy.shape)
    for row in range(char_numpy.shape[0]):
        for col in range(char_numpy.shape[1]):
            float_numpy[row][col] = float(char_numpy[row][col])
    print(float_numpy.shape)
    print(float_numpy)
    #方法二：创建一个嵌套的list
    float_result = [[] for i in range(3)]
    list_i = 0
    for row in result_row:
        float_result[list_i] = (list(map(float,row)))
        list_i += 1
    print(float_list)

    rows = len(result_row) #绘图的列数
    draw_pos = 1
    for i in float_numpy:
        plt.subplot(1,rows,draw_pos)
        draw_pos += 1
        #创建与每一行数据维数相同的点的数量作为横坐标
        x = np.linspace(0,500,500)
        plt.plot(x,i)
    plt.show()
'''











#函数封装区域
#读取原txt振动数据
def read_txt(path,colPos):
    f = codecs.open(path, mode='r', encoding='utf-8')
    # 前两行是表头，跳过读取
    line = f.readline()
    line = f.readline()
    # 第一行数据
    line = f.readline()
    listdata = []
    while line:
        a = line.split('\t')
        #读取第colPos列的数据
        col = a[colPos:colPos+1]
        listdata.append(col)
        line = f.readline()
    #读取的数据是string型，转化为float型
    #创建一个float型的list
    float_list = []
    for i in listdata:
        float_list.append(float(i[0]))
    #print(float_list)
    return float_list

#分割txt，取其中某些行
def split_txt(list1,begrow,endrow):
    list_res = list1[begrow:endrow]
    return list_res


#显示该txt中总体数据
def show_txt(list1):
    x = np.linspace(1, len(list1), len(list1))
    plt.plot(x, list1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('X方向垂直位移')
    plt.xlabel('points')
    plt.ylabel('ax(g)')
    plt.show()


#对数据集进行平移，后面的转到前面：
def translation_dataset(Data,transnumber):
    res_data = Data[len(Data) - transnumber:]
    res_data = np.append(res_data,Data[:len(Data) - transnumber])
    return res_data
#对数据集进行切片，移动

def cutin_trans_dataset(Data,cutin_numbers,cutin_size,trans_number):
    #begpos是被切分的每组数据的开始位置
    #print(Data)
    begpos = [None] * cutin_size
    i = 0
    while i < cutin_size:
        if i == 0:
            begpos[i] = random.randint(0,len(Data)-cutin_numbers)
            i += 1
        else:
            begpos[i] = random.randint(0,len(Data)-cutin_numbers)
            for j in range(i):
                if begpos[j] in range(begpos[i],begpos[i]+cutin_numbers):
                    i-=1
                    break
            i += 1
    print(begpos)
    numpy_cut = np.empty((cutin_size,cutin_numbers))
    for i in range(numpy_cut.shape[0]):
        numpy_cut[i] = np.array(Data[begpos[i] : begpos[i] + cutin_numbers])
        #print(Data[begpos[i] : begpos[i] + cutin_numbers])
    random.shuffle(begpos)
    print(begpos)
    for i in range(cutin_size):
        Data[begpos[i] : begpos[i]+cutin_numbers] = numpy_cut[i]
        #print(Data[begpos[i] : begpos[i]+cutin_numbers])
    #print(Data)
    resdata = translation_dataset(Data,trans_number)
    return resdata













#根据一段振动信号，制作一维CNN数据集，写入csv文件中
#相当一个滑动窗口，每次移动一个步长，将窗口内的数据作为一次振动类型的数据
def write_CSV(Data,csv_file_name,datanumbers,interval):
    with open(csv_file_name,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        beg_pos = 0
        #每次取出datanumbers个振动数据
        #对Data链表进行数据增强
        #Data = translation_dataset(Data,525)
        #切片平移处理法
        #Data = cutin_trans_dataset(Data,300,4,80)
        show_txt(Data)
        Data = split_txt(Data,400,600)
        Data = translation_dataset(Data,180)
        while beg_pos+datanumbers <= len(Data):
            list_datanumbers = Data[beg_pos:beg_pos+datanumbers]
            writer.writerow(list_datanumbers)
            #beg_pos每次移动的步长
            beg_pos+=interval
    csvfile.close()



def read_csv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        char_list = list(reader)
        # 方法一：list转numpy（推荐：方便后续深度学习）
        char_numpy = np.array(char_list)
        float_numpy = np.zeros(char_numpy.shape)
        for row in range(char_numpy.shape[0]):
            for col in range(char_numpy.shape[1]):
                float_numpy[row][col] = float(char_numpy[row][col])
        #print(float_numpy.shape)
        #print(float_numpy)
    return float_numpy



#绘制每一段振动
def draw_csv(numpy1):
    rows = numpy1.shape[0]  # 绘图的列数
    draw_pos = 1
    for i in numpy1:
        plt.subplot(1, rows, draw_pos)
        draw_pos += 1
        # 创建与每一行数据维数相同的点的数量作为横坐标
        x = np.linspace(0, numpy1.shape[1], numpy1.shape[1])
        plt.plot(x, i)
    plt.show()

#xlsx转csv
def xlsx_to_csv(xlsx_path):
    data_xlsx = pd.read_excel(xlsx_path,'Sheet1',index_col= 0)
    data_xlsx.to_csv('train.csv',encoding='utf-8')

#打乱原csv文件
def create_newcsv(path):
    lists = pd.read_csv(path,sep=r"\t",header=None)
    lists = lists.sample(frac=1)
    lists.to_csv('newtrain.csv',index=0)
    print('Finish save csv')

'''
#封装函数测试
#1.读取txt数据
list_float = read_txt('./201023180724.txt',2)
#2.选取所需的振动区域
list_split1 = split_txt(list_float,20,700)
print(len(list_split1))
#3.绘制该区域振动波形图
show_txt(list_split1)

write_CSV(list_float,'trainTest.csv',200,50)
numpy_float = read_csv('trainTest.csv')
print(numpy_float.shape)
draw_csv(numpy_float)
'''



#训练部分


root_url = "./train.csv"
data = read_csv(root_url)
data = np.array(data)
#print(x_train)
X = data[:,:data.shape[1]-1]
y = data[:,data.shape[1]-1:data.shape[1]]
y = y.astype(int)
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=50)
print(len(x_train.shape))
print(y_train.shape)
#print(x_train)

'''
nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

# transform the labels from integers to one hot vectors
enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

# save orignal y because later we will use binary
y_true = np.argmax(y_test, axis=1)

if len(x_train.shape) == 2:  # if univariate
    # add a dimension to make it multivariate with one dimension
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

input_shape = x_train.shape[1:]

acnn = Classifier_ENCODER(output_directory="./model",input_shape=input_shape,nb_classes=nb_classes)
acnn.build_model(input_shape=input_shape,nb_classes=nb_classes)
acnn.fit(x_train, y_train, x_test, y_test, y_true)
res = acnn.predict(x_test,y_true,x_train,y_train,y_test,False)
print(res)
print(y_test)
'''

'''
#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
'''
if __name__ == '__main__':

    #数据集准备
    xlsx_path = './train.xlsx'
    csv_path = './train.csv'
    #xlsx_to_csv(xlsx_path)
    data = read_csv(csv_path)

    print(data)
    print(data.shape)
    np.random.shuffle(data)
    print(data)
    #260为当前数据量总数
    Long = 260
    #200为训练集和验证集的分离点
    Lens = 180
    Batch_Size = 1

    def convert2oneHot(index,Lens):
        hot = np.zeros((Lens,))
        hot[int(index)] = 1
        return hot

    def xs_gen(numpy_floatdataset,batch_size = Batch_Size,train = True,Lens = Lens):
        if train:
            train_dataset = numpy_floatdataset[:Lens]
            print('find %s train dataset'% len(train_dataset))
            print('numpy1 is ',train_dataset[0,-1])
            steps = math.ceil(len(train_dataset)/batch_size)

        else:
            train_dataset = numpy_floatdataset[Lens:]
            print('find %s test dataset'% len(train_dataset))
            print('list 1 is ',train_dataset[0,-1])
            steps = math.ceil(len(train_dataset)/batch_size)

        while True:
            for i in range(steps):
                batch_numpy = numpy_floatdataset[i * batch_size: i * batch_size + batch_size]
                np.random.shuffle(batch_numpy)
                batch_x = np.array([file for file in batch_numpy[:,:-1]])
                batch_y = np.array([convert2oneHot(label,2) for label in batch_numpy[:,-1]])

                yield batch_x,batch_y


    def approx_res(res):
        approx_result = np.argmax(res,axis=1)
        return approx_result


    #convert2oneHot(1,2)
    #显示其中一条数据
    show_iter = xs_gen(data)
    for x,y in show_iter:
        x1 = x[0]
        y1 = y[0]
        print(x1.shape)
        print(y1)
        break

    plt.plot(x1)
    plt.show()


    #网络模型的搭建
    TIME_PERIODS = 200
    def build_cnn_model(inputshape = (TIME_PERIODS,),num_classes = 2):
        #model = Sequential()
        #input_shape = Reshape((TIME_PERIODS, 1), input_shape=inputshape)
        input_layer = keras.layers.Input((TIME_PERIODS,1))
        conv1 = Conv1D(16,8,strides=2,activation = 'relu',padding = 'same')(input_layer)
        #conv1 = Conv1D(16,8,strides=2,activation = 'relu',padding = 'same')(conv1)
        conv1 = MaxPooling1D(2,data_format='channels_last')(conv1)

        conv2 = Conv1D(32,4,strides=2,activation = 'relu',padding = 'same')(conv1)
        #conv2 = Conv1D(32,4,strides=2,activation = 'relu',padding = 'same')(conv2)
        conv2 = MaxPooling1D(2, data_format='channels_last')(conv2)

        conv3 = Conv1D(64, 2, strides=1, activation='relu', padding="same")(conv2)
        #conv3 = Conv1D(64, 2, strides=1, activation='relu', padding="same")(conv3)


        #attention_data = keras.layers.Lambda(lambda x: x[:, :, :32])(conv3)
        #attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 32:])(conv3)
        # attention mechanism
        #attention_softmax = keras.layers.Softmax()(attention_softmax)
        #multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])

        #globalpool = GlobalAveragePooling1D()(attention_softmax)
        dropout = Dropout(0.2)(conv3)
        flatten = Flatten()(dropout)
        fc_layer = Dense(96,activation='relu')(flatten)
        output_layer = Dense(num_classes, activation='softmax')(fc_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        return model

    def build_model(inputshape = (TIME_PERIODS,),num_classes = 2):
        #model = Sequential()
        #input_shape = Reshape((TIME_PERIODS, 1), input_shape=inputshape)
        input_layer = keras.layers.Input((TIME_PERIODS,1))
        conv1 = Conv1D(16,8,strides=2,activation = 'relu',padding = 'same')(input_layer)
        conv1 = Conv1D(16,8,strides=2,activation = 'relu',padding = 'same')(conv1)
        conv1 = MaxPooling1D(2,data_format='channels_last')(conv1)

        conv2 = Conv1D(32,4,strides=2,activation = 'relu',padding = 'same')(conv1)
        conv2 = Conv1D(32,4,strides=2,activation = 'relu',padding = 'same')(conv2)
        conv2 = MaxPooling1D(2, data_format='channels_last')(conv2)

        conv3 = Conv1D(64, 2, strides=1, activation='relu', padding="same")(conv2)
        conv3 = Conv1D(64, 2, strides=1, activation='relu', padding="same")(conv3)


        attention_data = keras.layers.Lambda(lambda x: x[:, :, :32])(conv3)
        attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 32:])(conv3)
        # attention mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])

        #globalpool = GlobalAveragePooling1D()(attention_softmax)
        dropout = Dropout(0.2)(multiply_layer)
        flatten = Flatten()(dropout)
        output_layer = Dense(num_classes, activation='softmax')(flatten)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        return model




    model = build_model()
    print(model.summary())

    #训练模式
    '''
    train_iter = xs_gen(data)
    val_iter = xs_gen(data,train=False)
    #训练参数的设置尚未理解，应结合其他教程学习基础知识
    ckpt = keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.4f}.h5',
        monitor='val_loss', save_best_only=True, verbose=1)
    opt = Adam(0.0002)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    history = model.fit_generator(
        generator=train_iter,
        steps_per_epoch=Lens // Batch_Size,
        epochs=50,
        initial_epoch=0,
        validation_data=val_iter,
        validation_steps=(Long - Lens) // Batch_Size,
        callbacks=[ckpt],
    )
    '''
    '''
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc.jpg')
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.jpg')
    '''

    #加载模型
    model = load_model('finishModel.h5')
    count = 0

    #此为读取一个生成器中的元素，与Batch_size大小有关
    '''
    for x,y in train_iter:
        plt.plot(x[0])
        plt.show()
        #print(x[0])
        new_x0 = np.expand_dims(x[0],axis=1)
        print(new_x0.shape)
        train_predict = model.predict(new_x0.T)
        print(train_predict)
        approx_predict = approx_res(train_predict)
        print(approx_predict)
        if(approx_predict[0] == 1):
            print("减速带产生的振动")
        print(y)
    '''

    '''
    for data_one in data:
        x = data_one[0:data_one.shape[0]-1]
        y = data_one[data_one.shape[0]-1:data_one.shape[0]]
        new_x0 = np.expand_dims(x, axis=1)
        print(new_x0.shape)
        train_predict = model.predict(new_x0.T)
        print(train_predict)
        approx_predict = approx_res(train_predict)
        #预测结果输出
        print(approx_predict)
        #实际标签输出
        print(y)
        if (approx_predict == 1):
            print("减速带产生的振动")
        plt.plot(x)
        plt.show()
    '''





    test_csv_path = './test.csv'
    # xlsx_to_csv(xlsx_path)
    test_data = read_csv(test_csv_path)
    x_test = test_data[:, :test_data.shape[1] - 1]
    y_test = test_data[:, test_data.shape[1] - 1:test_data.shape[1]]
    print(x_test[0].shape)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    #测试集也要转化为One-Hot编码格式，因为predict输出的也是One-Hot编码格式
    y_test = np.array([convert2oneHot(label,2) for label in y_test[:,-1]])
    #print(x_test.shape)
    #print(y_test.shape)
    #y_pred = model.predict(x_test, batch_size=1)
    #print(y_pred.shape)
    starttime = time.time()
    x_test0 = x_test[0].reshape((1,x_test.shape[1]))
    prediction = model.predict(x_test0)
    print(prediction)
    score = model.evaluate(x_test, y_test,batch_size=1)
    endtime = time.time()
    dtime = endtime - starttime
    print('Attention CNN Time:',dtime)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


    get_last_conv = keras.backend.function([model.layers[0].input], [model.layers[-8].output])
    last_conv = get_last_conv([x_test[:100], 1])[0]
    print((last_conv.shape))
    get_softmax = keras.backend.function([model.layers[0].input], [model.layers[-1].output])
    softmax = get_softmax([x_test[:100], 1])[0]
    softmax_weight = model.get_weights()[-2]
    print(softmax_weight.shape)
    CAM = np.dot(last_conv, softmax_weight)
    print(CAM.shape)
    y_test = y_test.reshape((y_test.shape[0],1))
    print(x_test[:100].shape)
# pp = PdfPages('CAM.pdf')
    for k in range(100):
        CAM = (CAM - CAM.min(axis=1, keepdims=True)) / (CAM.max(axis=1, keepdims=True) - CAM.min(axis=1, keepdims=True))
        c = np.exp(CAM) / np.sum(np.exp(CAM), axis=1, keepdims=True)
        print(c.shape)
        plt.figure(figsize=(13, 7));
        plt.plot(x_test[k].squeeze());
        plt.scatter(np.arange(len(x_test[k])), x_test[k].squeeze(), cmap='hot_r', c=c[ k, :, :, int(y_test[k])], s=100);
        plt.title(
            'True label:' + str(y_test[k]) + '   likelihood of label ' + str(y_test[k]) + ': ' + str(softmax[k][int(y_test[k])]))
        #plt.colorbar();
        plt.show()
#     pp.savefig()
#
# pp.close()
