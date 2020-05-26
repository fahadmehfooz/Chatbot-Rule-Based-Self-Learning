import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
import scipy.stats as stats
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.layers import Conv2D,MaxPool2D
from tensorflow.keras.optimizers import Adam


columns=['user','activity','time','x','y','z']


def fetch_train_data(accelorgyro,phoneorwatch):
    df=pd.DataFrame()
    for i in range(20):
        if i<10:
            dftemp=pd.read_csv(f'/content/drive/My Drive/Colab Notebooks/a/raw/train/{phoneorwatch}/{accelorgyro}/data_160{i}_{accelorgyro}_{phoneorwatch}.txt')
        else:
            dftemp=pd.read_csv(f'/content/drive/My Drive/Colab Notebooks/a/raw/train/{phoneorwatch}/{accelorgyro}/data_16{i}_{accelorgyro}_{phoneorwatch}.txt')

        if len(df)>0:
            df = pd.DataFrame( np.concatenate( (df.values, dftemp.values), axis=0 ) )
            df.columns=columns
        else:
            df=dftemp 
    return df


df_train_accel_phone=fetch_train_data('accel','phone')
df_train_accel_watch=fetch_train_data('accel','watch')
df_train_gyro_phone=fetch_train_data('gyro','phone')
df_train_gyro_watch=fetch_train_data('gyro','watch')

#merging training dataframes into single dataframe dftrain

dftrain = pd.DataFrame( np.concatenate( (df_train_accel_phone.values, df_train_accel_watch.values,df_train_gyro_phone,df_train_gyro_watch), axis=0 ) )
dftrain.columns=columns
dftrain.shape


def fetch_test_data(accelorgyro,phoneorwatch):
    df=pd.DataFrame()
    for i in range(20,34):
        dftemp=pd.read_csv(f'/content/drive/My Drive/Colab Notebooks/a/raw/test/{phoneorwatch}/{accelorgyro}/data_16{i}_{accelorgyro}_{phoneorwatch}.txt')

        if len(df)>0:
            df = pd.DataFrame( np.concatenate( (df.values, dftemp.values), axis=0 ) )
            df.columns=columns
        else:
            df=dftemp 
    return df

df_test_accel_phone=fetch_test_data('accel','phone')
df_test_accel_watch=fetch_test_data('accel','watch')
df_test_gyro_phone=fetch_test_data('gyro','phone')
df_test_gyro_watch=fetch_test_data('gyro','watch')

#merging testing dataframes into single dataframe dftest

dftest = pd.DataFrame( np.concatenate( (df_test_accel_phone.values, df_test_accel_watch.values,df_test_gyro_phone,df_test_gyro_watch), axis=0 ) )
dftest.columns=columns
dftest.shape

#removing semicolon from 'z' column and changing datatypes of 'x','y','z' to float as they have changed to object

dftrain['time']=dftrain['time'].astype(int)
dftest['time']=dftest['time'].astype(int)
dftrain['z'] = dftrain['z'].str.replace(';', '')
dftrain['z']=dftrain['z'].astype(float)
dftrain['x']=dftrain['x'].astype(float)
dftrain['y']=dftrain['y'].astype(float)
dftrain['time']=dftrain['time'].astype(int)
dftest['z'] = dftest['z'].str.replace(';', '')
dftest['z']=dftest['z'].astype(float)
dftest['y']=dftest['y'].astype(float)
dftest['x']=dftest['x'].astype(float)

#making a new dataframe without user and time, also getting activity count so that we can balance the dataset

df1=dftrain.drop(['user','time'],axis=1).copy()
df1['activity'].value_counts()


#Making new dataframes with respect to each activity to make a balanced training dataframe balanced


A=df1[df1['activity']=='A'].head(289076).copy()
B=df1[df1['activity']=='B'].head(289076).copy()
C=df1[df1['activity']=='C'].head(289076).copy()
D=df1[df1['activity']=='D'].head(289076).copy()
E=df1[df1['activity']=='E'].head(289076).copy()
F=df1[df1['activity']=='F'].head(289076).copy()
G=df1[df1['activity']=='G'].head(289076).copy()
H=df1[df1['activity']=='H'].head(289076).copy()
I=df1[df1['activity']=='I'].head(289076).copy()
J=df1[df1['activity']=='J'].head(289076).copy()
K=df1[df1['activity']=='K'].head(289076).copy()
L=df1[df1['activity']=='L'].head(289076).copy()
M=df1[df1['activity']=='M'].head(289076).copy()
N=df1[df1['activity']=='N'].head(289076).copy()
O=df1[df1['activity']=='O'].head(289076).copy()
P=df1[df1['activity']=='P'].head(289076).copy()
Q=df1[df1['activity']=='Q'].head(289076).copy()
R=df1[df1['activity']=='R'].head(289076).copy()
S=df1[df1['activity']=='S'].head(289076).copy()
balanced=pd.DataFrame()
balanced=balanced.append([A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S])
balanced['activity'].value_counts()

#making a new dataframe without user and time, also getting activity count so that we can balance the dataset

dft1=dftest.drop(['user','time'],axis=1).copy()
dft1
dft1['activity'].value_counts()

#Making new dataframes with respect to each activity to make a balanced testing dataframe balanced

A=dft1[dft1['activity']=='A'].head(223312).copy()
B=dft1[dft1['activity']=='B'].head(223312).copy()
C=dft1[dft1['activity']=='C'].head(223312).copy()
D=dft1[dft1['activity']=='D'].head(223312).copy()
E=dft1[dft1['activity']=='E'].head(223312).copy()
F=dft1[dft1['activity']=='F'].head(223312).copy()
G=dft1[dft1['activity']=='G'].head(223312).copy()
H=dft1[dft1['activity']=='H'].head(223312).copy()
I=dft1[dft1['activity']=='I'].head(223312).copy()
J=dft1[dft1['activity']=='J'].head(223312).copy()
K=dft1[dft1['activity']=='K'].head(223312).copy()
L=dft1[dft1['activity']=='L'].head(223312).copy()
M=dft1[dft1['activity']=='M'].head(223312).copy()
N=dft1[dft1['activity']=='N'].head(223312).copy()
O=dft1[dft1['activity']=='O'].head(223312).copy()
P=dft1[dft1['activity']=='P'].head(223312).copy()
Q=dft1[dft1['activity']=='Q'].head(223312).copy()
R=dft1[dft1['activity']=='R'].head(223312).copy()
S=dft1[dft1['activity']=='S'].head(223312).copy()
balancedt=pd.DataFrame()
balancedt=balancedt.append([A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S])
balancedt['activity'].value_counts()

#putting labels on activities for both balanced training dataframe and balanced testing dataframe

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()

balanced['Label']=label.fit_transform(balanced['activity'])
balanced

balancedt['Label']=label.fit_transform(balancedt['activity'])
balancedt

#Normalizing the data to get it in range of (-1,1)

X=balanced[['x','y','z']]
Y=balanced['Label']
scaler=StandardScaler()
X=scaler.fit_transform(X)
scaled_x=pd.DataFrame(data=X,columns=['x','y','z'])
scaled_x['Label']=Y.values
scaled_x


#Doing same for test dataframe

Xt=balancedt[['x','y','z']]
Yt=balancedt['Label']
Xt=scaler.fit_transform(Xt)
scaled_xt=pd.DataFrame(data=Xt,columns=['x','y','z'])
scaled_xt['Label']=Yt.values
scaled_xt

rate=20
frame_size=rate*4
hop_size=rate*2

#getting frames

def get_frames(df,frame_size,hop_size):
    n_features=3
    frames=[]
    labels=[]

    for i in range(0,len(df)-frame_size,hop_size):
        x=df['x'].values[i:i+frame_size]
        y=df['y'].values[i:i+frame_size]
        z=df['z'].values[i:i+frame_size]

        label=stats.mode(df['Label'][i:i+frame_size])[0][0]
        frames.append([x,y,z])
        labels.append(label)
    frames=np.asarray(frames).reshape(-1,frame_size,n_features)
    labels=np.asarray(labels)
    return frames,labels

X,y=get_frames(scaled_x,frame_size,hop_size)
Xt,yt=get_frames(scaled_xt,frame_size,hop_size)

#making data 3 diamensional for neural netwrk

x_train=X
y_train=y
print(x_train.shape)
x_test=Xt
y_test=yt
print(x_test.shape)

x_train=x_train.reshape(130083,80,3,1)
x_train[0].shape
x_test=x_test.reshape(100489,80,3,1)
x_test[0].shape

model=Sequential()
model.add(Conv2D(16,(2,2),activation='relu',input_shape=x_train[0].shape))
model.add(Dropout(0.1))

model.add(Conv2D(32,(2,2),activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(18,activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.002),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
BATCH_SIZE = 400
EPOCHS = 50

history = model.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      validation_data=(x_test,y_test),
                      verbose=1)


train_acc = model.evaluate(x_train, y_train, verbose=1)
test_acc = model.evaluate(x_test, y_test, verbose=1)