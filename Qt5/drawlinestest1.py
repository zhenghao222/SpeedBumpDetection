import array
import serial
import threading
import numpy as np
import time
import pyqtgraph as pg
from graphiclayout import Ui_MainWindow
import graphiclayout
from PyQt5.QtWidgets import QMainWindow,QWidget,QApplication
from PyQt5 import QtWidgets


import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import sklearn
import os
import csv
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from attention import Attention

# import some PyQt5 Image modules
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer

# gps module
from coord_convert.transform import wgs2gcj, wgs2bd, gcj2wgs, gcj2bd, bd2wgs, bd2gcj
import io
import folium
from folium import *

# import Opencv module
import cv2

'''
class MainWindow(QMainWindow):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
           
        # create a timer
        self.timer1 = QTimer()
        # set timer timeout callback function
        self.timer1.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.ui.pushButton.clicked.connect(self.controlTimer)

    # view camera
    def viewCam(self):
        # read image in BGR format
        ret, image = self.cap.read()
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_left = image[:,0:1280]
        image_right = image[:,1280:2560]
        #cv2.imshow("image_left",image_left)
        #cv2.imshow("image_right",image_right)
        # get image infos
        height, width, channel = image_left.shape
        step = channel * width
        #print("height = ",height)
        #print("width= ",width)
        #print("step = ",step)
        # create QImage from image
        qImg_left = QImage(image_left.data.tobytes(), width, height, step, QImage.Format_RGB888)
        qImg_right = QImage(image_right.data.tobytes(), width, height, step, QImage.Format_RGB888)

        qImg_left = qImg_left.scaled(self.ui.label.width(), self.ui.label.height())
        qImg_right = qImg_right.scaled(self.ui.label_2.width(), self.ui.label_2.height())

        # show image in img_label
        self.ui.label_2.setPixmap(QPixmap.fromImage(qImg_left))
        self.ui.label_3.setPixmap(QPixmap.fromImage(qImg_right))


    # start/stop timer1
    def controlTimer():
        # if timer is stopped
        if not timer1.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(1)
            #??????????????????
            self.cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc('M','J','P','G'))
            self.cap.set(cv2.CAP_PROP_FPS, 60);
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,2560)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,960)
            # start timer1
            self.timer1.start(33)
            # update control_bt text
            self.ui.pushButton.setText("Stop")
            # if timer1 is started
        else:
            # stop timer1
            self.timer1.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.ui.pushButton.setText("Start")

'''

def viewCam():
    # read image in BGR format
    ret, image = cap.read()
    # convert image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_left = image[:,0:320]
    image_right = image[:,320:640]
    #cv2.imshow("image_left",image_left)
    #cv2.imshow("image_right",image_right)
    # get image infos
    height, width, channel = image_left.shape
    step = channel * width
    #print("height = ",height)
    #print("width= ",width)
    #print("step = ",step)
    # create QImage from image
    qImg_left = QImage(image_left.data.tobytes(), width, height, step, QImage.Format_RGB888)
    qImg_right = QImage(image_right.data.tobytes(), width, height, step, QImage.Format_RGB888)

    qImg_left = qImg_left.scaled(ui.label_2.width(), ui.label_2.height())
    qImg_right = qImg_right.scaled(ui.label_3.width(), ui.label_3.height())

    # show image in img_label
    ui.label_2.setPixmap(QPixmap.fromImage(qImg_left))
    ui.label_3.setPixmap(QPixmap.fromImage(qImg_right))


# start/stop timer1
def controlTimer():
    # if timer is stopped
    global cap
    if not timer1.isActive():
        # create video capture
        #??????????????????
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FPS, 60);
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
        # start timer1
        timer1.start(30)
        #th_camera.start()
        # update control_bt text
        ui.pushButton.setText("Stop")
    # if timer1 is started
    else:
        # stop timer1
        timer1.stop()
        # release video capture
        cap.release()
        # update control_bt text
        ui.pushButton.setText("Start")



event = threading.Event()
model_data = 0;

ACCData = [0.0] * 8
GYROData = [0.0] * 8
AngleData = [0.0] * 8  # ??????????????????????????????????????????????????????????????????

FrameState = 0  # ??????0x???????????????????????????????????????
Bytenum = 0  # ??????????????????????????????
CheckSum = 0  # ???????????????

def DueData(inputdata):  # ???????????????????????????????????????????????????????????????????????????????????????
    global FrameState  # ???????????????????????????????????????global?????????
    global Bytenum
    global CheckSum
    for data in inputdata:  # ??????????????????????????????
        if FrameState == 0:  # ????????????????????????????????????????????????
            if data == 0x55 and Bytenum == 0:  # 0x55???????????????????????????????????????????????????bytenum
                CheckSum = data
                Bytenum = 1
                continue
            elif data == 0x51 and Bytenum == 1:  # ???byte??????0 ??? ????????? 0x51 ??????????????????frame
                CheckSum += data
                FrameState = 1
                Bytenum = 2
            elif data == 0x52 and Bytenum == 1:  # ??????
                CheckSum += data
                FrameState = 2
                Bytenum = 2
            elif data == 0x53 and Bytenum == 1:
                CheckSum += data
                FrameState = 3
                Bytenum = 2
        elif FrameState == 1:  # acc    #??????????????????????????????
            if Bytenum < 10:  # ??????8?????????
                ACCData[Bytenum - 2] = data  # ???0??????
                CheckSum += data
                Bytenum += 1
            else:
                if data == (CheckSum & 0xff):  # ?????????????????????
                    acc_x, acc_y, acc_z = get_acc(ACCData)
                CheckSum = 0  # ??????????????????????????????????????????
                Bytenum = 0
                FrameState = 0
        elif FrameState == 2:  # gyro
            if Bytenum < 10:
                GYROData[Bytenum - 2] = data
                CheckSum += data
                Bytenum += 1
            else:
                if data == (CheckSum & 0xff):
                   get_gyro(GYROData)
                CheckSum = 0
                Bytenum = 0
                FrameState = 0
        elif FrameState == 3:  # angle
            if Bytenum < 10:
                AngleData[Bytenum - 2] = data
                CheckSum += data
                Bytenum += 1
            else:
                if data == (CheckSum & 0xff):
                    get_angle(AngleData)
                CheckSum = 0
                Bytenum = 0
                FrameState = 0
    return acc_x

def get_acc(datahex):  # ?????????
    axl = datahex[0]
    axh = datahex[1]
    ayl = datahex[2]
    ayh = datahex[3]
    azl = datahex[4]
    azh = datahex[5]

    k_acc = 16

    acc_x = (axh << 8 | axl) / 32768 * k_acc
    acc_y = (ayh << 8 | ayl) / 32768 * k_acc
    acc_z = (azh << 8 | azl) / 32768 * k_acc
    if acc_x >= k_acc:
        acc_x -= 2 * k_acc
    if acc_y >= k_acc:
        acc_y -= 2 * k_acc
    if acc_z >= k_acc:
        acc_z -= 2 * k_acc

    return acc_x, acc_y, acc_z

def get_gyro(datahex):  # ?????????
    wxl = datahex[0]
    wxh = datahex[1]
    wyl = datahex[2]
    wyh = datahex[3]
    wzl = datahex[4]
    wzh = datahex[5]
    k_gyro = 2000

    gyro_x = (wxh << 8 | wxl) / 32768 * k_gyro
    gyro_y = (wyh << 8 | wyl) / 32768 * k_gyro
    gyro_z = (wzh << 8 | wzl) / 32768 * k_gyro
    if gyro_x >= k_gyro:
        gyro_x -= 2 * k_gyro
    if gyro_y >= k_gyro:
        gyro_y -= 2 * k_gyro
    if gyro_z >= k_gyro:
        gyro_z -= 2 * k_gyro
    return gyro_x, gyro_y, gyro_z

def get_angle(datahex):  # ??????
    rxl = datahex[0]
    rxh = datahex[1]
    ryl = datahex[2]
    ryh = datahex[3]
    rzl = datahex[4]
    rzh = datahex[5]
    k_angle = 180

    angle_x = (rxh << 8 | rxl) / 32768 * k_angle
    angle_y = (ryh << 8 | ryl) / 32768 * k_angle
    angle_z = (rzh << 8 | rzl) / 32768 * k_angle
    if angle_x >= k_angle:
        angle_x -= 2 * k_angle
    if angle_y >= k_angle:
        angle_y -= 2 * k_angle
    if angle_z >= k_angle:
        angle_z -= 2 * k_angle

    return angle_x, angle_y, angle_z

model_data = 0
i = 0
number = 0
countsum = 0

def Serial():
    while (True):
        n = mSerial.inWaiting()
        if (n):
            if data != " ":
                datahex = mSerial.read(33)  # ??????hex()??????????????????read???????????????16??????
                acc_x = DueData(datahex)  # ????????????????????????,?????????x?????????????????????
                n = 0
                global countsum
                global i;
                global number
                global model_data
                if i < historyLength:
                    number += 1
                    data[i] = acc_x
                    i = i + 1
                else:
                    data[:-1] = data[1:]
                    data[i - 1] = acc_x
                    number += 1
                if number == 200:
                    number = 0
                    model_data = np.array(data)
                    # print("model data", model_data)
                    event.set()
                    countsum += 1
                    # print(countsum)

def plotData():
    curve.setData(data)

TIME_PERIODS = 200

def build_model(inputshape=(TIME_PERIODS,), num_classes=2):
    # model = Sequential()
    # input_shape = Reshape((TIME_PERIODS, 1), input_shape=inputshape)
    input_layer = keras.layers.Input((TIME_PERIODS, 1))
    conv1 = Conv1D(16, 8, strides=2, activation='relu', padding='same')(input_layer)
    conv1 = Conv1D(16, 8, strides=2, activation='relu', padding='same')(conv1)
    conv1 = MaxPooling1D(2, data_format='channels_last')(conv1)

    conv2 = Conv1D(32, 4, strides=2, activation='relu', padding='same')(conv1)
    conv2 = Conv1D(32, 4, strides=2, activation='relu', padding='same')(conv2)
    conv2 = MaxPooling1D(2, data_format='channels_last')(conv2)

    conv3 = Conv1D(64, 2, strides=1, activation='relu', padding="same")(conv2)
    conv3 = Conv1D(64, 2, strides=1, activation='relu', padding="same")(conv3)

    attention_data = keras.layers.Lambda(lambda x: x[:, :, :32])(conv3)
    attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 32:])(conv3)
    # attention mechanism
    attention_softmax = keras.layers.Softmax()(attention_softmax)
    multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])

    # globalpool = GlobalAveragePooling1D()(attention_softmax)
    dropout = Dropout(0.2)(multiply_layer)
    flatten = Flatten()(dropout)
    output_layer = Dense(num_classes, activation='softmax')(flatten)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    return model

def predictionspeedbump():
    while True:
        event.wait()
        global model_data
        model_input = model_data.reshape((1, model_data.shape[0]))
        prediction = model.predict(model_input)
        print("????????????", prediction)
        if (np.argmax(prediction) == 1):
            ui.lineEdit.setText("?????????")
        else:
            ui.lineEdit.setText("????????????")
        event.clear()

def thread2test():
    while True:
        event.wait()
        print("the second thread!")
        event.clear()

#??????????????????
def expandShipments(row, column):
    item = ui.tableWidget.item(row, column)
    print(item.text())
    string_text = item.text()
    lon, lat = string_text.split(',')
    lat = float(lat)
    lon = float(lon)
    gcj_lon, gcj_lat = wgs2gcj(lon, lat)
    local = folium.Map(location=[gcj_lat, gcj_lon], zoom_start=20,
                       tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
                       attr='default')

    folium.Marker([gcj_lat, gcj_lon],
                  popup=str('Home'),
                  icon=folium.Icon(color="blue", icon="info-sign"),
                  ).add_to(local)

    local.save("local.html")
    gpsdata = io.BytesIO()
    local.save(gpsdata, close_file=False)
    ui.webEngineView.setHtml(gpsdata.getvalue().decode())
    ui.webEngineView.show()
    mw.show()

if __name__ == "__main__":
    app = pg.mkQApp()  # ??????app
    mw = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mw)

    # create a timer
    timer1 = QTimer()
    # set timer timeout callback function
    timer1.timeout.connect(viewCam)
    # set control_bt callback clicked  function
    ui.pushButton.clicked.connect(controlTimer)
    cap = cv2.VideoCapture(1)

    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    pw = pg.PlotWidget()
    pw.setRange(xRange=[0,200])
    ui.verticalLayout.addWidget(pw)
    #??????????????????
    #win = pg.GraphicsWindow()  # ????????????
    #win.setWindowTitle(u'pyqtgraph??????????????????')
    #win.resize(800, 500)  # ???????????????
    #data = array.array('i')  # ??????????????????????????????,double?????????
    historyLength = 200  # ???????????????
    a = 0
    data=np.zeros(historyLength).__array__('d')#????????????????????????
    #p = win.addPlot()  # ??????p??????????????????
    #p.showGrid(x=True, y=True)  # ???X???Y???????????????
    #p.setRange(xRange=[0, historyLength], yRange=[-6, 6], padding=0)
    #p.setLabel(axis='left', text='y / V')  # ??????
    #p.setLabel(axis='bottom', text='x / point')
    #p.setTitle('semg')  # ???????????????
    curve = pw.plot()  # ??????????????????

    ui.lineEdit.setReadOnly(True)



    portx = 'COM7'
    bps = 115200
    # ?????????????????????????????? ??????open???????????????
    mSerial = serial.Serial(portx, int(bps))
    model = build_model()
    print(model.summary())
    model = load_model('finishModel.h5')
    if (mSerial.isOpen()):
        print("open success")
        #mSerial.write("hello".encode()) # ?????????????????? ?????????????????????
        mSerial.flushInput()  # ???????????????
    else:
        print("open failed")
        serial.close()  # ????????????
    th1 = threading.Thread(target=Serial)#??????????????????????????????????????????BUG????????????
    th2 = threading.Thread(target=predictionspeedbump)
    th1.start()
    th2.start()
    timer = pg.QtCore.QTimer()
    timer.timeout.connect(plotData)  # ????????????????????????
    timer.start(33)  # ??????ms????????????

    #gps??????

    lon, lat = 109.200389667, 34.3749975
    gcj_lon, gcj_lat = wgs2gcj(lon, lat)
    local = folium.Map(location=[gcj_lat, gcj_lon], zoom_start=20,
                       tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
                       attr='default')

    # ?????????????????????
    folium.Marker([gcj_lat, gcj_lon],
                  popup=str('Home'),
                  icon=folium.Icon(color="red", icon="info-sign"),
                  ).add_to(local)
    #local.save("local.html")
    gpsdata = io.BytesIO()
    local.save(gpsdata, close_file=False)
    ui.webEngineView.setHtml(gpsdata.getvalue().decode())
    ui.webEngineView.show()

    #????????????
    ui.tableWidget.setRowCount(4)
    ui.tableWidget.setColumnCount(1)
    ui.tableWidget.setHorizontalHeaderLabels(['speedbump geographical location information '])
    ui.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    ui.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    ui.tableWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

    newItem = QtWidgets.QTableWidgetItem('109.200389667, 34.3749975')
    ui.tableWidget.setItem(0, 0, newItem)
    newItem = QtWidgets.QTableWidgetItem('109.2263011, 34.372329')
    ui.tableWidget.setItem(1, 0, newItem)
    newItem = QtWidgets.QTableWidgetItem('109.2561756, 34.370358')
    ui.tableWidget.setItem(2, 0, newItem)
    #????????????????????????
    ui.tableWidget.cellDoubleClicked.connect(expandShipments)

    mw.show()
    app.exec_()