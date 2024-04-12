import os
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load, dump

def LoadMSData(root_path, data_path, t = False):
    data1 = pd.read_csv(os.path.join(root_path, data_path),header=None)
    dfx = pd.DataFrame(data1).T if t else pd.DataFrame(data1)         #是否转置
    feas = list(dfx.iloc[0])                                          #提取第一行核质比信息
    feas = list(map(lambda x:str(x),feas))                            #特征只能是整数或字符串
    dfx = dfx.drop(dfx.index[0])                                      #原核质比行可以清除
    dfx.index = list(range(len(dfx)))                                 #行索引重新编号
    dfx.columns = feas                                                #特征值存为列名
    dfx = dfx.astype(float)
    return dfx

def PlotCurve(History):
    keys = History.history.keys()
    if 'loss' in keys:
        loss = History.history['loss']
        plt.plot(loss, label='loss')
    if 'val_loss' in keys:
        val_loss = History.history['val_loss']
        plt.plot(val_loss, label='val_loss')
    if 'accuracy' in keys:
        Acc = History.history['accuracy']
        plt.plot(Acc, label='accuracy')
    if 'val_accuracy' in keys:
        val_Acc = History.history['val_accuracy']
        plt.plot(val_Acc, label='val_accuracy')
    if 'auc' in keys:
        auc = History.history['auc']
        plt.plot(auc, label='auc')
    if 'val_auc' in keys:
        val_auc = History.history['val_auc']
        plt.plot(val_auc, label='val_auc')
    plt.legend(loc='upper right')
    plt.show()

def SaveMpOrX(data, save_path, data_num, file_type, file_name):
    path = os.path.join(save_path, data_num, file_type)
    if not os.path.exists(path):
        os.makedirs(path)
    dump(data, os.path.join(path, file_name))