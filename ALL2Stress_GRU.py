#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[13]:


import tensorflow as tf
import tensorflow.keras as tfkeras
#%matplotlib inline
import pandas as pd
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from itertools import product
import os

#多特征输入；多特征预测；多步长预测
#num_argin=40#输入:输出==__:,
num_delay=0
num_argout=1
#num_batsiz=16#batch_size的值,
num_feat=7  #输入特征数量  
num_featout=1 #输出特征数为1
#num_times=1 #重复验证训练次数
num_epochs = 100
K_valid=4
perval=6  #24/4
#std_ae;R2 done
#测试集画预测对比图，验证集只验证指标不画预测情况图 done
#多步长预测时，分离每步的预测相对误差率 ->
#写函数封装，输入各个样本数据集，得到各个参数再加入到总集统计中

#list_batsiz=[16,32,64] #48,80,80,96,128]
#list_redfac=[0.5,0.1,0.05] #[0.8,0.5,0.3,0.1,0.05]
#list_numlay1=[16,32,64,] #48,80,96,112,128]
#list_numlay2=[16,32,64,] #48,80,96,112,128]
#list_nargin=[10,20,30,40,50]  #,60,70,80]
#num of layers;patience
num_batsiz,redfac,num_argin=128,0.5,10
num_times_tt=5

#数据读取
nameset=[]
for filename in os.listdir('data\\stress_strain'):
    nameset.append(filename.split('.')[0])

#namey1=nameset
namey1=[111,112,113,114,121,122,123,124,212,213,214,221,222,223,311,312,313,321,322,323,324,333,334,424,]#,331,613,634
datatconya0=np.zeros((1))
datatconya1=np.zeros((1))
datatconya2=np.zeros((1))
datatconya3=np.zeros((1))
datatconya4=np.zeros((1))
datatconya5=np.zeros((1))
datatconya6=np.zeros((1))

#datatconrtp=np.zeros((1))
#datatconrtpn=np.zeros((1))
mean_=np.zeros((len(namey1),num_feat))
std_=np.zeros((len(namey1),num_feat))
#mean_out=np.zeros((len(namey1),num_argout))
#std_out=np.zeros((len(namey1),num_argout))
len_per_datcsvhit=np.zeros((1)) 
for i in range(len(namey1)):
    a0='data\\para-time_prediction\\{0:s}stress.csv'
    b0=a0.format(str(namey1[i]))
    data0=pd.read_csv(b0)
    dat0=data0.dropna(axis=0,how='any')
    dat0=dat0.iloc[:,-1].values[:dat0.shape[0]]
    
    lendat=dat0.shape[0]
    datn0=dat0[:lendat]
    len_per_datcsvhit=np.append(len_per_datcsvhit,np.array([lendat]),axis=0)
    #lenrtp=np.argmax(dat0)
    #datn0=dat0[:lenrtp+1] 
    #datrtp=0.2*np.arange(lenrtp,-1,-1)
    #len_per_datcsvhit=np.append(len_per_datcsvhit,np.array([lenrtp+1]),axis=0)
    mean_[i,0]=datn0.mean(axis=0)            #各输入特征标准化 
    datn0-=mean_[i,0]
    std_[i,0]=datn0.std(axis=0)
    datn0/=std_[i,0]
    #datrtpn=datrtp.copy()
    #mean_out[i,0]=datrtpn.mean(axis=0)            #各输出特征标准化 
    #datrtpn-=mean_out[i,0]
    #std_out[i,0]=datrtpn.std(axis=0)
    #datrtpn/=std_out[i,0]
    datatconya0=np.append(datatconya0,datn0,axis=0) 
    #datatconrtp=np.append(datatconrtp,datrtp,axis=0)
    #datatconrtpn=np.append(datatconrtpn,datrtpn,axis=0)
    
    a1='data\\para-time_prediction\\{0:s}hit.txt'
    b1=a1.format(str(namey1[i]))
    dat1=pd.read_csv(b1,sep='\s+')
    datn1=dat1.values[:lendat,1]
    mean_[i,1]=datn1.mean(axis=0)            #各输入特征标准化 
    datn1-=mean_[i,1]
    std_[i,1]=datn1.std(axis=0)
    datn1/=std_[i,1]
    datatconya1=np.append(datatconya1,datn1,axis=0) 
    
    a2='data\\para-time_prediction\\{0:s}hit_rate.txt'
    b2=a2.format(str(namey1[i]))
    dat2=pd.read_csv(b2,sep='\s+')
    datn2=dat2.values[:lendat,1]
    mean_[i,2]=datn2.mean(axis=0)            #各输入特征标准化 
    datn2-=mean_[i,2]
    std_[i,2]=datn2.std(axis=0)
    datn2/=std_[i,2]
    datatconya2=np.append(datatconya2,datn2,axis=0)
    
    a3='data\\para-time_prediction\\{0:s}energy.txt'
    b3=a3.format(str(namey1[i]))
    dat3=pd.read_csv(b3,sep='\s+')
    datn3=dat3.values[:lendat,1]
    mean_[i,3]=datn3.mean(axis=0)            #各输入特征标准化 
    datn3-=mean_[i,3]
    std_[i,3]=datn3.std(axis=0)
    datn3/=std_[i,3]
    datatconya3=np.append(datatconya3,datn3,axis=0)
    
    a4='data\\para-time_prediction\\{0:s}energy_rate.txt'
    b4=a4.format(str(namey1[i]))
    dat4=pd.read_csv(b4,sep='\s+')
    datn4=dat4.values[:lendat,1]
    mean_[i,4]=datn4.mean(axis=0)            #各输入特征标准化 
    datn4-=mean_[i,4]
    std_[i,4]=datn4.std(axis=0)
    datn4/=std_[i,4]
    datatconya4=np.append(datatconya4,datn4,axis=0)
    
    a5='data\\para-time_prediction\\{0:s}ASL.txt'
    b5=a5.format(str(namey1[i]))
    dat5=pd.read_csv(b5,sep='\s+')
    datn5=dat5.values[:lendat,1]
    mean_[i,5]=datn5.mean(axis=0)            #各输入特征标准化 
    datn5-=mean_[i,5]
    std_[i,5]=datn5.std(axis=0)
    datn5/=std_[i,5]
    datatconya5=np.append(datatconya5,datn5,axis=0)
    
    a6='data\\para-time_prediction\\{0:s}RMS.txt'
    b6=a6.format(str(namey1[i]))
    dat6=pd.read_csv(b6,sep='\s+')
    datn6=dat6.values[:lendat,1]
    mean_[i,6]=datn6.mean(axis=0)            #各输入特征标准化 
    datn6-=mean_[i,6]
    std_[i,6]=datn6.std(axis=0)
    datn6/=std_[i,6]
    datatconya6=np.append(datatconya6,datn6,axis=0)
        
len_cumsum_datcsvhit=len_per_datcsvhit.cumsum()      #对时长列表进行累计求和
len_per_datcsvhit=len_per_datcsvhit.astype(np.int32)
len_cumsum_datcsvhit=len_cumsum_datcsvhit.astype(np.int32)

print("导入训练数据的总集合维数： ",datatconya1.shape)
fig=plt.figure()           #画图
plt.rcParams['axes.linewidth'] = 1  # 图框宽度
plt.rcParams['figure.dpi'] = 300  # plt.show显示分辨率
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 15}
plt.rc('font', **font)
plt.plot(range(len(datatconya0)-1),datatconya0[1:],'r')
plt.show()
plt.plot(range(len(datatconya1)-1),datatconya1[1:],'b')
plt.show()
plt.plot(range(len(datatconya2)-1),datatconya2[1:],'y')
plt.show()
plt.plot(range(len(datatconya3)-1),datatconya3[1:],'k')
plt.show()
plt.plot(range(len(datatconya4)-1),datatconya4[1:],'g')
plt.show()
plt.plot(range(len(datatconya5)-1),datatconya5[1:],'cyan')
plt.show()
plt.plot(range(len(datatconya6)-1),datatconya6[1:],'salmon')
plt.show()

datatconya0=np.reshape(datatconya0,(datatconya0.shape[0],1))
datatconya1=np.reshape(datatconya1,(datatconya1.shape[0],1))
datatconya2=np.reshape(datatconya2,(datatconya2.shape[0],1))
datatconya3=np.reshape(datatconya3,(datatconya3.shape[0],1))
datatconya4=np.reshape(datatconya4,(datatconya4.shape[0],1))
datatconya5=np.reshape(datatconya5,(datatconya5.shape[0],1))
datatconya6=np.reshape(datatconya6,(datatconya6.shape[0],1))
datatconya=np.concatenate((datatconya0,datatconya1,datatconya2,datatconya3,
                         datatconya4,datatconya5,datatconya6),axis=1)

#构建训练集转化至模型输入的形式
x_train_dat,y_train_dat=[],[]
for l in range(len(len_cumsum_datcsvhit)-1):
    for i in range(num_argin+num_delay+1+len_cumsum_datcsvhit[l],
                   len_cumsum_datcsvhit[l+1]+1-(num_argout-1)):#此处加2是考虑下降点，加1是不考虑，也就无法训练断裂结束的状态
        x_train_dat.append(datatconya[i-num_argin-num_delay:i-num_delay,:])
        y_train_dat.append(datatconya0[i:i+num_argout])                             #形成前15/50个-->16th/51st的输入数据       
x_train_dat,y_train_dat=np.array(x_train_dat),np.array(y_train_dat)          #列表转化为矩阵
x_train_dat=np.reshape(x_train_dat,(x_train_dat.shape[0],x_train_dat.shape[1],num_feat))
y_train_dat=np.reshape(y_train_dat,(y_train_dat.shape[0],y_train_dat.shape[1],1))
x_train_data,y_train_data=np.zeros((1,num_argin,num_feat)),np.zeros((1,num_argout,1)) #本试验np初始因声发射不同样本，具有特殊性
x_train_data=np.append(x_train_data,x_train_dat,axis=0)
y_train_data=np.append(y_train_data,y_train_dat,axis=0)
print('训练集的输入、输出维数： ',x_train_data.shape,y_train_data.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_data, y_train_data))
train_dataset = train_dataset.batch(num_batsiz)

pcc_test=[]
r2_test=[]
r2_test1=[]
mae_test=[]
mae_test1=[]
mae_test2=[]
mae_test3=[]
mae_test4=[]
mae_test5=[]
rmse_test=[]
rmse_test1=[]
rmse_test2=[]
rmse_test3=[]
rmse_test4=[]
rmse_test5=[]

mape_test1=[]
#mape_test2=[]
mape_test2=[]
mape_test21=[]
mape_test22=[]
mape_test23=[]
mape_test24=[]
mape_test25=[]
mape_test3=[]
mape_test31=[]
mape_test32=[]
mape_test33=[]
mape_test34=[]
mape_test35=[]
sum_time_train=[]
average_time_train=[]
sum_time_predict=[]
average_time_predict=[]
#mape_all_histories=[]
for t in range(num_times_tt):
    print('第',t+1,'次重复性训练与测试：')
    ##训练
    all_loss=[]
    all_mae_history = []
    train_time_perset=[]
    print('正在训练...')
    # Build the Keras model 
    start1=time.time()
    modelt = models.Sequential()
    modelt.add(layers.GRU(128, activation='relu',return_sequences=True,
                          input_shape=(x_train_data.shape[1],num_feat),
                         kernel_initializer='he_normal'))#x_train_data.shape[1]   ###he_uniform
    #model.add(layers.Dense(16, activation='relu'))
    modelt.add(layers.Dropout(0.1))
    modelt.add(layers.GRU(64, activation='relu',return_sequences=True, kernel_initializer='he_normal'))
    modelt.add(layers.Dropout(0.1))
    modelt.add(layers.GRU(64, activation='relu', kernel_initializer='he_normal'))
    modelt.add(layers.Dropout(0.1))
    #modelt.add(layers.Dense(64, activation='relu', kernel_initializer='he_normal'))
    modelt.add(layers.Dense(num_argout, ))
    
    str_nbs=str(num_batsiz)
    str_rf=str(redfac)
    str_nagi=str(num_argin)
    str_t=str(t+1)
    savedname='Model_saved/'
    folder = os.path.exists(savedname)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(savedname) 
    savep=savedname+'model_1012_shahreaeraslrms2stress_{0:s}_{1:s}_{2:s}_{3:s}'
    savepathf=savep.format(str_nbs,str_rf,str_nagi,str_t)
    callbacks_list=[
        tfkeras.callbacks.EarlyStopping(monitor='mae',patience=20,),
        tfkeras.callbacks.ModelCheckpoint(
            filepath=savepathf+'.h5',
            monitor='mae',
            save_best_only=True,
            verbose=1,
            mode="min",
        ),
        tfkeras.callbacks.ReduceLROnPlateau(monitor='mae',factor=redfac,min_lr=0.00001,
                                            patience=10,verbose=1,)]
    modelt.compile(optimizer='rmsprop', loss='mse', metrics=['mape','mae'])
    
    # Train the model (in silent mode(静默模式), verbose=0)
    history = modelt.fit(train_dataset,
                        callbacks=callbacks_list,
                        epochs=num_epochs,verbose=1)#,batch_size=num_batsiz
    end1=time.time()
    print('训练结束。') 
    plot_model(modelt,show_shapes=True,to_file=savepathf+'.png')
    print('已保存模型及结构图')
    
    train_time_perset.append(end1-start1)
    all_loss = history.history['loss']
    all_mae_history = history.history['mae']
    
    ##画图（epochs-loss;epochs-mae）
    print('训练损失最小步数： ',np.argmin(all_loss))
    plt.plot(range(1, len(all_loss) + 1), all_loss, 'b', label='Training loss')
    plt.title('Training Loss(In the Complete Training Dataset)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    print('训练MAE最小步数： ',np.argmin(all_mae_history))
    plt.plot(range(1, len(all_mae_history)+1), all_mae_history)
    plt.title('Mean Absolute Error(In the Complete Training Dataset)')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.show()
    def smooth_curve(points, factor=0.9):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points
    smooth_mae_history = smooth_curve(all_mae_history)
    print('训练MAE指数移动平滑化后的最小步数： ',np.argmin(smooth_mae_history))
    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history,'r')
    plt.title('Smooth Mean Absolute Error(In the Complete Training Dataset)')
    plt.xlabel('Epochs')
    plt.ylabel('Smooth MAE(In the complete training dataset)')
    plt.show()
    epoch_test=np.argmin(all_mae_history)
    if epoch_test==0:
        epoch_test=0.1
        
    ##测试
    #testInd=[3,10,14,18,22]
    nameyt=[533,534,613,634,622,632]
    mean_test=np.zeros((len(nameyt),num_feat))
    std_test=np.zeros((len(nameyt),num_feat))
    #mean_testout=np.zeros((len(nameyt),num_argout))
    #std_testout=np.zeros((len(nameyt),num_argout))
    testdataya0=np.zeros((1))
    testdataya1=np.zeros((1))
    testdataya2=np.zeros((1))
    testdataya3=np.zeros((1))
    testdataya4=np.zeros((1))
    testdataya5=np.zeros((1))
    testdataya6=np.zeros((1))
    #testdatartp=np.zeros((1))
    #testdatartpn=np.zeros((1))
    testdasiz=np.zeros((1))
    for j in range(len(nameyt)):
        aa0='data\\para-time_prediction\\{0:s}stress.csv'
        bb0=aa0.format(str(nameyt[j]))
        datat0=pd.read_csv(bb0)#,sep='\s+',,data_test
        datt0=datat0.dropna(axis=0,how='any')
        datt0=datt0.iloc[:,-1].values         #dataframe-->numpy.ndarray类型//
        #lentrtp=np.argmax(datt1)
        #testdasiz=np.append(testdasiz,np.array([lentrtp+1]),axis=0)        #添加时长
        #dattrtp=0.2*np.arange(lentrtp,-1,-1)
        #dattn1=datt1[:lentrtp+1]
        lentdat=datt0.shape[0]
        testdasiz=np.append(testdasiz,np.array([lentdat]),axis=0)        #添加时长
        dattn0=datt0[:lentdat]
        
        mean_test[j,0]=dattn0.mean(axis=0)             #各输入特征标准化
        dattn0-=mean_test[j,0]
        std_test[j,0]=dattn0.std(axis=0)
        dattn0/=std_test[j,0]
        #dattrtpn=dattrtp.copy()
        #mean_testout[j,0]=dattrtpn.mean(axis=0)             #各输出特征标准化
        #dattrtpn-=mean_testout[j,0]
        #std_testout[j,0]=dattrtpn.std(axis=0)
        #dattrtpn/=std_testout[j,0]
        testdataya0=np.append(testdataya0,dattn0,axis=0)    #赋值连接样本矩阵#np.array()
        #testdatartp=np.append(testdatartp,dattrtp,axis=0)#np.array()
        #testdatartpn=np.append(testdatartpn,dattrtpn,axis=0)
        
        aa1='data\\para-time_prediction\\{0:s}hit.txt'
        bb1=aa1.format(str(nameyt[j]))
        datt1=pd.read_csv(bb1,sep='\s+')
        dattn1=datt1.values[:lentdat,1]
        mean_test[j,1]=dattn1.mean(axis=0)            #各输入特征标准化 
        dattn1-=mean_test[j,1]
        std_test[j,1]=dattn1.std(axis=0)
        dattn1/=std_test[j,1]
        testdataya1=np.append(testdataya1,dattn1,axis=0)
        
        aa2='data\\para-time_prediction\\{0:s}hit_rate.txt'
        bb2=aa2.format(str(nameyt[j]))
        datt2=pd.read_csv(bb2,sep='\s+')
        dattn2=datt2.values[:lentdat,1]
        mean_test[j,2]=dattn2.mean(axis=0)            #各输入特征标准化 
        dattn2-=mean_test[j,2]
        std_test[j,2]=dattn2.std(axis=0)
        dattn2/=std_test[j,2]
        testdataya2=np.append(testdataya2,dattn2,axis=0)
        
        aa3='data\\para-time_prediction\\{0:s}energy.txt'
        bb3=aa3.format(str(nameyt[j]))
        datt3=pd.read_csv(bb3,sep='\s+')
        dattn3=datt3.values[:lentdat,1]
        mean_test[j,3]=dattn3.mean(axis=0)            #各输入特征标准化 
        dattn3-=mean_test[j,3]
        std_test[j,3]=dattn3.std(axis=0)
        dattn3/=std_test[j,3]
        testdataya3=np.append(testdataya3,dattn3,axis=0)
        
        aa4='data\\para-time_prediction\\{0:s}energy_rate.txt'
        bb4=aa4.format(str(nameyt[j]))
        datt4=pd.read_csv(bb4,sep='\s+')
        dattn4=datt4.values[:lentdat,1]
        mean_test[j,4]=dattn4.mean(axis=0)            #各输入特征标准化 
        dattn4-=mean_test[j,4]
        std_test[j,4]=dattn4.std(axis=0)
        dattn4/=std_test[j,4]
        testdataya4=np.append(testdataya4,dattn4,axis=0)
        
        aa5='data\\para-time_prediction\\{0:s}ASL.txt'
        bb5=aa5.format(str(nameyt[j]))
        datt5=pd.read_csv(bb5,sep='\s+')
        dattn5=datt5.values[:lentdat,1]
        mean_test[j,5]=dattn5.mean(axis=0)            #各输入特征标准化 
        dattn5-=mean_test[j,5]
        std_test[j,5]=dattn5.std(axis=0)
        dattn5/=std_test[j,5]
        testdataya5=np.append(testdataya5,dattn5,axis=0)
        
        aa6='data\\para-time_prediction\\{0:s}RMS.txt'
        bb6=aa6.format(str(nameyt[j]))
        datt6=pd.read_csv(bb6,sep='\s+')
        dattn6=datt6.values[:lentdat,1]
        mean_test[j,6]=dattn6.mean(axis=0)            #各输入特征标准化 
        dattn6-=mean_test[j,6]
        std_test[j,6]=dattn6.std(axis=0)
        dattn6/=std_test[j,6]
        testdataya6=np.append(testdataya6,dattn6,axis=0)
    testdasizcum=testdasiz.cumsum()
    testdasizcum=testdasizcum.astype(np.int32)
    testdasiz=testdasiz.astype(np.int32)
    #testdatay3=testdataya3[:] #512
    #testdatay3=np.append(testdatay3,(0-mean_test[len(nameyt)-1])/std_test[len(nameyt)-1])
    
    print("导入测试数据的总集合维数： ",testdataya0.shape)
    fig=plt.figure()           #画图
    plt.plot(range(len(testdataya0)-1),testdataya0[1:],'r')
    plt.show()
    
    testdataya0=np.reshape(testdataya0,(testdataya0.shape[0],1))
    testdataya1=np.reshape(testdataya1,(testdataya1.shape[0],1))
    testdataya2=np.reshape(testdataya2,(testdataya2.shape[0],1))
    testdataya3=np.reshape(testdataya3,(testdataya3.shape[0],1))
    testdataya4=np.reshape(testdataya4,(testdataya4.shape[0],1))
    testdataya5=np.reshape(testdataya5,(testdataya5.shape[0],1))
    testdataya6=np.reshape(testdataya6,(testdataya6.shape[0],1))
    testdataya=np.concatenate((testdataya0,testdataya1,testdataya2,testdataya3,testdataya4,
                              testdataya5,testdataya6),axis=1)
    
    x_test_data,y_test_data=[],[]               
    for l in range(len(testdasizcum)-1):
        for n in range(num_argin+num_delay+1+testdasizcum[l],testdasizcum[l+1]+1-(num_argout-1)):
            x_test_data.append(testdataya[n-num_argin:n,:])
            y_test_data.append(testdataya0[n:n+num_argout])                             #形成前15个-->16th的输入数据
            
    x_test_data,y_test_data=np.array(x_test_data),np.array(y_test_data)          #列表转化为矩阵
    x_test_data=np.reshape(x_test_data,(x_test_data.shape[0],x_test_data.shape[1],num_feat))
    y_test_data=np.reshape(y_test_data,(y_test_data.shape[0],y_test_data.shape[1],1))
    print('测试集的输入输出维数： ',x_test_data.shape,y_test_data.shape)   #441,15
    
    print('开始测试：')
    
    all_rmse_test=[]
    all_rmse_test1=[]
    all_rmse_test2=[]
    all_rmse_test3=[]
    all_rmse_test4=[]
    all_rmse_test5=[]
    all_mae_test=[]
    all_mae_test1=[]
    all_mae_test2=[]
    all_mae_test3=[]
    all_mae_test4=[]
    all_mae_test5=[]
    all_mape_test=[]
    all_mape_test2=[]
    mape_all_history1=[]
    mape_all_history2=[]
    mape_all_history3=[]
    mape_all_history4=[]
    mape_all_history5=[]
    #mape_all_history=np.zeros((1,num_argout))
    #mape_all_history2=np.zeros((1,num_argout))
    #mape_all_history3=np.zeros((1,num_argout))
    #mape_all_history1=np.zeros((1,num_argout))
    pcc_alltest=[]
    r2_alltest=[]
    r2_alltest1=[]
    mape_test_new=[]
    mape_test_new1=[]
    mape_test_new2=[]
    mape_test_new3=[]
    mape_test_new4=[]
    mape_test_new5=[]
    testdatapredicted=np.zeros((1))
    predict_time_perset=[]
    for m in range(len(nameyt)):
        print('\nTestset #',m+1)
        # Build the Keras model
        modelttest = load_model(savepathf+'.h5')
        
        test_mse_score, test_mape_score,test_mae_score = modelttest.evaluate(
            x_test_data[testdasizcum[m]-m*(num_argin+num_argout-1+num_delay):
                        testdasizcum[m+1]-(m+1)*(num_argin+num_argout-1+num_delay)],
            y_test_data[testdasizcum[m]-m*(num_argin+num_argout-1+num_delay):
                        testdasizcum[m+1]-(m+1)*(num_argin+num_argout-1+num_delay)])
        all_mape_test.append(test_mape_score)
        all_mae_test.append(test_mae_score*std_test[m,0])
        
        all_rmse_test.append(np.sqrt(test_mse_score)*std_test[m,0])
        
        start2=time.time()
        y_predict=y_test_data.copy()
        y_rawm=y_test_data[testdasizcum[m]-m*(num_argin+num_argout-1+num_delay):
                        testdasizcum[m+1]-(m+1)*(num_argin+num_argout-1+num_delay)].copy()
        #print(y_predict.shape)
        print('单测试集原始数据的维数： ',y_rawm.shape)
        #y_predict[testdasizcum[m]-m*(num_argin+num_argout-1+num_delay):
        #          testdasizcum[m+1]-(m+1)*(num_argin+num_argout-1+num_delay)]=modelttest.predict(
        #    x_test_data[testdasizcum[m]-m*(num_argin+num_argout-1+num_delay):
        #        testdasizcum[m+1]-(m+1)*(num_argin+num_argout-1+num_delay)]).reshape(-1,num_argout,num_feat)
        y_predict=modelttest.predict(x_test_data).reshape(-1,num_argout,num_featout)
        y_predictm=modelttest.predict(x_test_data[testdasizcum[m]-m*(num_argin+num_argout-1+num_delay):
                        testdasizcum[m+1]-(m+1)*(num_argin+num_argout-1+num_delay)]).reshape(-1,num_argout,num_featout)
        #print(y_predict.shape)
        print('单测试集测试结果的维数： ',y_predictm.shape)
        end2=time.time()
        predict_time_perset.append(end2-start2)
        
        datapertest=np.append(np.zeros((num_argin)),y_predictm[:,0,0]*std_test[m,0]+mean_test[m,0])
        testdatapredicted=np.append(testdatapredicted,datapertest,axis=0)
        
        plt.plot(range(1, num_argin + 2), testdataya1[testdasizcum[m]+1:testdasizcum[m]+num_argin+2], 'aqua',)
        plt.plot(range( num_argin + 1, y_rawm.shape[0] +num_argin+ 1),y_rawm[: ,0,0]*std_test[m,0]+mean_test[m,0],
                 'aqua', label='raw')
        plt.plot(range( num_argin + 1, y_predictm.shape[0] +num_argin+ 1),y_predictm[:,0,0]*std_test[m,0]+mean_test[m,0],
                 'coral', label='predict')
        plt.axvline(x=num_argin+0.5,ls="--",c="skyblue")
        plt.title('Contrast between Ground truth value and Predicted')
        plt.xlabel('Time')
        plt.ylabel('Load/kN')
        plt.legend()
        plt.show()
        #plt.plot(range(1,num_argin+2),testdatartp[testdasizcum[m]+1:testdasizcum[m]+num_argin+2]*std_test[m]+mean_test[m],'aqua')
        #plt.plot(range(num_argin+1,y_rawm.shape[0]+num_argin+1),y_rawm[:,0,0]*std_test[m]+mean_test[m],'aqua',label='raw')
        #plt.plot(range(num_argin+1,y_predictm.shape[0]+num_argin+1),y_predictm[:,0,0]*std_test[m]+mean_test[m],'coral',label='predict')
        #plt.axvline(x=num_argin+0.5,ls="--",c="skyblue")
        #plt.title('Contrast after inverse normalization')
        #plt.xlabel('time order')
        #plt.ylabel('ylabel')
        #plt.legend()
        #plt.show()
        
        #绘图及计算MAE
        #time_index=np.arange(num_argout).reshape(num_argout,1)
        perpred_mape=[]
        perpred_raw=[]
        perpred_pred=[]
        #k=0
        for i in range(testdasizcum[m]-m*(
            num_argin+num_argout-1+num_delay),testdasizcum[m+1]-(m+1)*(
            num_argin+num_argout-1+num_delay)):
            y_pdraw=y_test_data[i].reshape(num_argout,1)
            y_pdpredict=y_predict[i]
            y_pdraw=y_pdraw*std_test[m,0]+mean_test[m,0]
            y_pdpredict=y_pdpredict*std_test[m,0]+mean_test[m,0]
            #print(y_pdraw,y_pdpredict)
            #print("test sample #  ",i+1)
            #plt.plot(time_index,y_pdraw,'aqua',label='raw')
            #plt.plot(time_index,y_pdpredict,'coral',label='predict')
            #plt.legend()
            #plt.xlabel('time/s')
            #plt.ylabel('Hit')
            #plt.show()
            perpred_raw.append(y_pdraw)
            #perpred_raw=np.reshape(perpred_raw,(-1,num_argout,num_feat))
            perpred_pred.append(y_pdpredict)
            #perpred_pred=np.reshape(perpred_pred,(-1,num_argout,num_feat))
            #perpred_mape.append(round(np.mean(abs(y_pdraw-y_pdpredict)/y_pdraw)*100,2))
            perpred_mape.append(abs(y_pdraw-y_pdpredict)/(y_pdraw+0.01)*100)
            #k+=1
            #print("该样本的平均相对误差率为：",round(np.mean(abs(y_pdraw-y_pdpredict)/y_pdraw)*100,2),"%")
            p=0
            for _ in y_pdraw:
                if _<0:
                    p=p+1
            if p>0: 
                print("     ！异常序号：n,m,i = ",n,m,i)
                print("            第",n+1,"次，第",m+1,"个测试样本，第",i+1,"次测试的平均相对误差率为：",
                      round(np.mean(abs(y_pdraw-y_pdpredict)/y_pdraw)*100,2),"%")
        #print('mape记录长度',len(perpred_mape))        #print(perpred_mape)
        #print(mape_all_history)
        mape_pertest_history=[]
        mape_pertest_history1=[]
        mape_pertest_history2=[]
        mape_pertest_history3=[]
        mape_pertest_history.append(perpred_mape)#[:-1]
        mape_pertest_history=np.array(mape_pertest_history)
        mape_pertest_history=np.reshape(mape_pertest_history,(-1,num_argout,num_feat))
        print('单测试集的MAPE记录的维度： ',mape_pertest_history.shape)
        print("standard deviation of mape: ",mape_pertest_history.std())
        plt.plot(range(1, len(mape_pertest_history[:,0,0])+1), mape_pertest_history[:,0,0], 'aqua',label='mape')
        #plt.plot(range(num_argin+1,y_predict.shape[0]+1),y_test_data[:,0,0]*std_test[m]+mean_test[m],'aqua',label='raw')
        #plt.plot(range(num_argin+1,y_predict.shape[0]+1),y_predict[:,0,0]*std_test[m]+mean_test[m],'coral',label='predict')
        #plt.axvline(x=num_argin+0.5,ls="--",c="skyblue")
        plt.title('Distribution of Mape')
        plt.xlabel('Predicted Time')
        plt.ylabel('MAPE')
        plt.legend()
        plt.show()
        mape_pertest_history1.append(perpred_mape[:-1])#[:-1]
        mape_pertest_history1=np.array(mape_pertest_history1)
        mape_pertest_history1=np.reshape(mape_pertest_history1,(-1,num_argout,num_feat))
        print('单测试集的MAPE(-1)记录的维度： ',mape_pertest_history1.shape)
        print("standard deviation of mape-1: ",mape_pertest_history1.std())
        plt.plot(range(1, len(mape_pertest_history1[:,0,0])+1), mape_pertest_history1[:,0,0], 'aqua',label='mape')
        #plt.plot(range(num_argin+1,y_predict.shape[0]+1),y_test_data[:,0,0]*std_test[m]+mean_test[m],'aqua',label='raw')
        #plt.plot(range(num_argin+1,y_predict.shape[0]+1),y_predict[:,0,0]*std_test[m]+mean_test[m],'coral',label='predict')
        #plt.axvline(x=num_argin+0.5,ls="--",c="skyblue")
        plt.title('Distribution of Mape')
        plt.xlabel('Predicted Time')
        plt.ylabel('MAPE')
        plt.legend()
        plt.show()
        mape_pertest_history2.append(perpred_mape[:-2])#[:-1]
        mape_pertest_history2=np.array(mape_pertest_history2)
        mape_pertest_history2=np.reshape(mape_pertest_history2,(-1,num_argout,num_feat))
        print('单测试集的MAPE(-2)记录的维度： ',mape_pertest_history2.shape)
        print("standard deviation of mape-2: ",mape_pertest_history2.std())
        plt.plot(range(1, len(mape_pertest_history2[:,0,0])+1), mape_pertest_history2[:,0,0], 'aqua',label='mape')
        #plt.plot(range(num_argin+1,y_predict.shape[0]+1),y_test_data[:,0,0]*std_test[m]+mean_test[m],'aqua',label='raw')
        #plt.plot(range(num_argin+1,y_predict.shape[0]+1),y_predict[:,0,0]*std_test[m]+mean_test[m],'coral',label='predict')
        #plt.axvline(x=num_argin+0.5,ls="--",c="skyblue")
        plt.title('Distribution of Mape')
        plt.xlabel('Predicted Time')
        plt.ylabel('MAPE')
        plt.legend()
        plt.show()
        mape_pertest_history3.append(perpred_mape[:-3])#
        mape_pertest_history3=np.array(mape_pertest_history3)
        mape_pertest_history3=np.reshape(mape_pertest_history3,(-1,num_argout,num_feat))
        print('单测试集的MAPE(-3)记录的维度： ',mape_pertest_history3.shape)
        print("standard deviation of mape-3: ",mape_pertest_history3.std())
        plt.plot(range(1, len(mape_pertest_history3[:,0,0])+1), mape_pertest_history3[:,0,0], 'aqua',label='mape')
        #plt.plot(range(num_argin+1,y_predict.shape[0]+1),y_test_data[:,0,0]*std_test[m]+mean_test[m],'aqua',label='raw')
        #plt.plot(range(num_argin+1,y_predict.shape[0]+1),y_predict[:,0,0]*std_test[m]+mean_test[m],'coral',label='predict')
        #plt.axvline(x=num_argin+0.5,ls="--",c="skyblue")
        plt.title('distribution of mape')
        plt.xlabel('predict time')
        plt.ylabel('mape')
        plt.legend()
        plt.show()
                
        #计算相似系数
        perpred_raw=np.array(perpred_raw)
        perpred_raw=np.reshape(perpred_raw,(-1,num_argout,num_feat))
        perpred_pred=np.array(perpred_pred)
        perpred_pred=np.reshape(perpred_pred,(-1,num_argout,num_feat))
        pcc=np.mean((perpred_raw[:,0,0]- perpred_raw[:,0,0].mean())*(
            perpred_pred[:,0,0]-perpred_pred[:,0,0].mean()))/(
            perpred_raw[:,0,0].std()*perpred_pred[:,0,0].std())
        print('单测试集的皮尔逊相似系数为：',pcc) 
        pcc_alltest.append(pcc)
        #计算决定系数
        r2=1-np.sum((perpred_pred[:,0,0]-perpred_raw[:,0,0])**2)/np.sum((perpred_raw[:,0,0]- perpred_raw[:,0,0].mean())**2)
        r21=np.sum((perpred_pred[:,0,0]-np.mean(perpred_raw[:,0,0]))**2)/np.sum((perpred_raw[:,0,0]- perpred_raw[:,0,0].mean())**2)
        print('r2,r21:',r2,r21)
        r2_adj=1-(1-r2)*(perpred_raw.shape[0]-1)/(perpred_raw.shape[0]-1-num_argin)
        r2_adj1=1-(1-r21)*(perpred_raw.shape[0]-1)/(perpred_raw.shape[0]-1-num_argin)
        r2_alltest.append(r2_adj)
        r2_alltest1.append(r2_adj1)
        #计算mae
        all_mae_test1.append(np.mean(abs(perpred_raw[:,0,0]-perpred_pred[:,0,0])[:-1]))
        all_mae_test2.append(np.mean(abs(perpred_raw[:,0,0]-perpred_pred[:,0,0])[:-2]))
        all_mae_test3.append(np.mean(abs(perpred_raw[:,0,0]-perpred_pred[:,0,0])[:-3]))
        all_mae_test4.append(np.mean(abs(perpred_raw[:,0,0]-perpred_pred[:,0,0])[:-4]))
        all_mae_test5.append(np.mean(abs(perpred_raw[:,0,0]-perpred_pred[:,0,0])[:-5]))
        #计算rmse
        all_rmse_test1.append(np.sqrt(np.mean(((perpred_raw[:,0,0]-perpred_pred[:,0,0])**2)[:-1])))
        all_rmse_test2.append(np.sqrt(np.mean(((perpred_raw[:,0,0]-perpred_pred[:,0,0])**2)[:-2])))
        all_rmse_test3.append(np.sqrt(np.mean(((perpred_raw[:,0,0]-perpred_pred[:,0,0])**2)[:-3])))
        all_rmse_test4.append(np.sqrt(np.mean(((perpred_raw[:,0,0]-perpred_pred[:,0,0])**2)[:-4])))
        all_rmse_test5.append(np.sqrt(np.mean(((perpred_raw[:,0,0]-perpred_pred[:,0,0])**2)[:-5])))
        #计算MAPE
        mape_test_new.append(round(np.mean(abs(perpred_raw-perpred_pred)/perpred_raw)*100,2))
        mape_test_new1.append(round(np.mean(abs(perpred_raw-perpred_pred)[:-1]/perpred_pred[:-1])*100,2))
        mape_test_new2.append(round(np.mean(abs(perpred_raw-perpred_pred)[:-2]/perpred_pred[:-2])*100,2))
        mape_test_new3.append(round(np.mean(abs(perpred_raw-perpred_pred)[:-3]/perpred_pred[:-3])*100,2))
        mape_test_new4.append(round(np.mean(abs(perpred_raw-perpred_pred)[:-4]/perpred_pred[:-4])*100,2))
        mape_test_new5.append(round(np.mean(abs(perpred_raw-perpred_pred)[:-5]/perpred_pred[:-5])*100,2))
        
        
        perpred_mape=np.array(perpred_mape)
        perpred_mape=np.reshape(perpred_mape,(-1,num_argout))
        #mape_all_history=np.append(mape_all_history,perpred_mape,axis=0)
        #print('mape单测试集递增记录维数：（行结果再减1）  ',mape_all_history.shape)
        mape_all_history1.append(np.mean(perpred_mape[:-1]))
        mape_all_history2.append(np.mean(perpred_mape[:-2]))
        mape_all_history3.append(np.mean(perpred_mape[:-3]))
        mape_all_history4.append(np.mean(perpred_mape[:-4]))
        mape_all_history5.append(np.mean(perpred_mape[:-5]))
        #mape_all_history1=np.append(mape_all_history1,perpred_mape[:-1],axis=0)
        #mape_all_history2=np.append(mape_all_history2,perpred_mape[:-2],axis=0)
        #mape_all_history3=np.append(mape_all_history3,perpred_mape[:-3],axis=0)
        #plt.plot(range(len(mape_all_history3)),mape_all_history3,'aqua',label='MAPE')
        #plt.title('distribution of mape_all_history3')
        #plt.xlabel('predict time')
        #plt.ylabel('mape')
        #plt.legend()
        #plt.show()
        all_mape_test2.append(round(np.mean(perpred_mape.flatten()),2))
    #mape_all_history.append(mape_all_history)    
    #print('mape汇总记录维数：（行结果再减1） ',mape_all_history.shape)                        ##多步长预测
    
    average_pcc_test=np.mean(pcc_alltest)
    average_r2_test=np.mean(r2_alltest)
    average_r2_test1=np.mean(r2_alltest1)
    
    average_mae_test = np.mean(np.array(all_mae_test))
    average_mae_test1 = np.mean(np.array(all_mae_test1))
    average_mae_test2 = np.mean(np.array(all_mae_test2))
    average_mae_test3 = np.mean(np.array(all_mae_test3))
    average_mae_test4 = np.mean(np.array(all_mae_test4))
    average_mae_test5 = np.mean(np.array(all_mae_test5))
    average_rmse_test=np.mean(all_rmse_test)
    average_rmse_test1=(np.mean(all_rmse_test1))
    average_rmse_test2=(np.mean(all_rmse_test2))
    average_rmse_test3=(np.mean(all_rmse_test3))
    average_rmse_test4=np.mean(all_rmse_test4)
    average_rmse_test5=np.mean(all_rmse_test5)
    
    average_mape_test = np.mean(all_mape_test)
    average_mape_test2= np.mean(all_mape_test2)
    average_mape_test21= np.mean(mape_all_history1)
    average_mape_test22= np.mean(mape_all_history2)
    average_mape_test23= np.mean(mape_all_history3)
    average_mape_test24= np.mean(mape_all_history4)
    average_mape_test25= np.mean(mape_all_history5)
    average_mape_test3= np.mean(mape_test_new)
    average_mape_test31= np.mean(mape_test_new1)
    average_mape_test32= np.mean(mape_test_new2)
    average_mape_test33= np.mean(mape_test_new3)
    average_mape_test34= np.mean(mape_test_new4)
    average_mape_test35= np.mean(mape_test_new5)
    print('    平均相似系数r2为：',round(average_pcc_test,2))
    print('    平均决定系数R2为：',round(average_r2_test,2))
    print('    平均决定系数R21为：',round(average_r2_test1,2))
    print('    平均绝对误差为：',round(average_mae_test,2))
    print('    平均绝对误差1为：',round(average_mae_test1,2))
    print('    平均绝对误差2为：',round(average_mae_test2,2))
    print('    平均绝对误差3为：',round(average_mae_test3,2))
    print('    平均绝对误差4为：',round(average_mae_test4,2))
    print('    平均绝对误差5为：',round(average_mae_test5,2))
    print('    根均方误差为  ：',round(average_rmse_test,2))
    print('    根均方误差1为  ：',round(average_rmse_test1,2))
    print('    根均方误差2为  ：',round(average_rmse_test2,2))
    print('    根均方误差3为  ：',round(average_rmse_test3,2))
    print('    根均方误差4为  ：',round(average_rmse_test4,2))
    print('    根均方误差5为  ：',round(average_rmse_test5,2))
    print('    平均绝对误差率1：',round(average_mape_test,2),'%')                            #evaluate
    #print('    平均绝对误差率2：',round(np.mean(all_mae_test*std_test/mean_test)*100,2),'%')#evaluate+calculate RTP未标准化
    print('    平均绝对误差率2：',round(average_mape_test2,2),'%')                           #calculate
    print('    平均绝对误差率21：',round(average_mape_test21,2),'%')
    print('    平均绝对误差率22：',round(average_mape_test22,2),'%')
    print('    平均绝对误差率23：',round(average_mape_test23,2),'%')
    print('    平均绝对误差率24：',round(average_mape_test24,2),'%') 
    print('    平均绝对误差率25：',round(average_mape_test25,2),'%') 
    average_train_time_perset=sum(train_time_perset)/epoch_test
    sum_train_time_perset=sum(train_time_perset)
    print('    训练网络用时:%.2f秒'%sum_train_time_perset,'\t训练网络平均每轮用时:%.2f秒/轮'%average_train_time_perset)
    average_predict_time_perset=np.mean(predict_time_perset)/epoch_test
    sum_predict_time_perset=np.mean(predict_time_perset)
    print('    测试网络用时:%.2f秒'%sum_predict_time_perset,'\t测试网络平均每轮用时:%.2f秒/轮'%average_predict_time_perset)
    pcc_test.append(round(average_pcc_test,2))
    r2_test.append(round(average_r2_test,2))
    r2_test1.append(round(average_r2_test1,2))
    mae_test.append(round(average_mae_test,2))
    mae_test1.append(round(average_mae_test1,2))
    mae_test2.append(round(average_mae_test2,2))
    mae_test3.append(round(average_mae_test3,2))
    mae_test4.append(round(average_mae_test4,2))
    mae_test5.append(round(average_mae_test5,2))
    rmse_test.append(round(average_rmse_test,2))
    rmse_test1.append(round(average_rmse_test1,2))
    rmse_test2.append(round(average_rmse_test2,2))
    rmse_test3.append(round(average_rmse_test3,2))
    rmse_test4.append(round(average_rmse_test4,2))
    rmse_test5.append(round(average_rmse_test5,2))
    
    mape_test1.append(round(average_mape_test,2))
    #mape_test2.append(round(np.mean(all_mae_test*std_test/mean_test)*100,2))
    mape_test2.append(round(average_mape_test2,2))
    mape_test21.append(round(average_mape_test21,2))
    mape_test22.append(round(average_mape_test22,2))
    mape_test23.append(round(average_mape_test23,2))
    mape_test24.append(round(average_mape_test24,2))
    mape_test25.append(round(average_mape_test25,2))
    mape_test3.append(round(average_mape_test3,2))
    mape_test31.append(round(average_mape_test31,2))
    mape_test32.append(round(average_mape_test32,2))
    mape_test33.append(round(average_mape_test33,2))
    mape_test34.append(round(average_mape_test34,2))
    mape_test35.append(round(average_mape_test35,2))
    sum_time_train.append(sum_train_time_perset)
    average_time_train.append(average_train_time_perset)
    sum_time_predict.append(sum_predict_time_perset)
    average_time_predict.append(average_predict_time_perset)
    
    print("全部测试集预测对比图： ")
    fig=plt.figure()           #画图
    plt.plot(range(len(testdataya1)-1),testdataya1[1:],'b-.',label='Ground truth value')
    plt.plot(range(len(testdatapredicted)-1),testdatapredicted[1:],'r',label='Predicted value')
    plt.title('Contrast between Ground truth value and Predicted')
    #for i in range(len(testdasizcum)-1):
    #    plt.axvline(x=testdasizcum[i]+num_argin+0.5,ls="-.",c="coral")
    plt.xlabel('Timestamps')
    plt.ylabel('Load/kN')
    plt.legend()
    plt.show()
    
print("\n本测试组的超参数配置为：batchsize=",num_batsiz,"  reduce factor=",redfac,'  numn of argin=',num_argin)
print('平均相似系数r2分别为：    ',pcc_test,'   平均相似系数r2的平均值为：',round(np.mean(pcc_test),2))
print('平均决定系数R2分别为：    ',r2_test,'   平均决定系数R2的平均值为：',round(np.mean(r2_test),2))
print('平均决定系数R21分别为：    ',r2_test1,'   平均决定系数R2的平均值为：',round(np.mean(r2_test1),2))
print('平均绝对误差分别为： ',mae_test,'   平均绝对误差的平均值为：',round(np.mean(mae_test),2))
print('平均绝对误差1分别为： ',mae_test1,'   平均绝对误差1的平均值为：',round(np.mean(mae_test1),2))
print('平均绝对误差2分别为： ',mae_test2,'   平均绝对误差2的平均值为：',round(np.mean(mae_test2),2))
print('平均绝对误差3分别为： ',mae_test3,'   平均绝对误差3的平均值为：',round(np.mean(mae_test3),2))
print('平均绝对误差4分别为： ',mae_test4,'   平均绝对误差4的平均值为：',round(np.mean(mae_test4),2))
print('平均绝对误差5分别为： ',mae_test5,'   平均绝对误差5的平均值为：',round(np.mean(mae_test5),2))
print('根均方误差分别为： ',rmse_test,'   根均方误差的平均值为：',round(np.mean(rmse_test),2))
print('根均方误差1分别为： ',rmse_test1,'   根均方误差1的平均值为：',round(np.mean(rmse_test1),2))
print('根均方误差2分别为： ',rmse_test2,'   根均方误差2的平均值为：',round(np.mean(rmse_test2),2))
print('根均方误差3分别为： ',rmse_test3,'   根均方误差3的平均值为：',round(np.mean(rmse_test3),2))
print('根均方误差4分别为： ',rmse_test4,'   根均方误差4的平均值为：',round(np.mean(rmse_test4),2))
print('根均方误差5分别为： ',rmse_test5,'   根均方误差5的平均值为：',round(np.mean(rmse_test5),2))
print('平均绝对误差率1分别为： ',mape_test1,'   平均绝对误差率1的平均值为：',round(np.mean(mape_test1),2))
#print('平均绝对误差率2分别为： ',mape_test2,'   平均绝对误差率2的平均值为：',round(np.mean(mape_test2),2))
print('平均绝对误差率2分别为： ',mape_test2,'   平均绝对误差率2的平均值为：',round(np.mean(mape_test2),2))
print('平均绝对误差率21分别为： ',mape_test21,'   平均绝对误差率21的平均值为：',round(np.mean(mape_test21),2))
print('平均绝对误差率22分别为： ',mape_test22,'   平均绝对误差率22的平均值为：',round(np.mean(mape_test22),2))
print('平均绝对误差率23分别为： ',mape_test23,'   平均绝对误差率23的平均值为：',round(np.mean(mape_test23),2))
print('平均绝对误差率24分别为： ',mape_test24,'   平均绝对误差率24的平均值为：',round(np.mean(mape_test24),2))
print('平均绝对误差率25分别为： ',mape_test25,'   平均绝对误差率25的平均值为：',round(np.mean(mape_test25),2))
print('平均绝对误差率3分别为： ',mape_test3,'   平均绝对误差率3的平均值为：',round(np.mean(mape_test3),2))
print('平均绝对误差率31分别为： ',mape_test31,'   平均绝对误差率31的平均值为：',round(np.mean(mape_test31),2))
print('平均绝对误差率32分别为： ',mape_test32,'   平均绝对误差率32的平均值为：',round(np.mean(mape_test32),2))
print('平均绝对误差率33分别为： ',mape_test33,'   平均绝对误差率33的平均值为：',round(np.mean(mape_test33),2))
print('平均绝对误差率34分别为： ',mape_test34,'   平均绝对误差率34的平均值为：',round(np.mean(mape_test34),2))
print('平均绝对误差率35分别为： ',mape_test35,'   平均绝对误差率35的平均值为：',round(np.mean(mape_test35),2))
print('训练网络用时分别为: ',sum_time_train,'   训练网络用时的平均值为：',round(np.mean(sum_time_train),2))
print('训练网络平均每轮用时:(秒/轮)',average_time_train,'   训练网络平均每轮用时的平均值为：',round(np.mean(average_time_train),2))
print('测试网络用时分别为:',sum_time_predict,'   测试网络用时的平均值为：',round(np.mean(sum_time_predict),2))
print('测试网络平均每轮用时:(秒/轮)',average_time_predict,'   测试网络平均每轮用时的平均值为：',round(np.mean(average_time_predict),2)) 


# In[ ]:




