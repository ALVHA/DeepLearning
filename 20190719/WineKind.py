
# coding: utf-8

# ### 와인종류 예측하기
# 
# * 샘플의 수 : 6496개
# 
# 
# * 속성 : 12개
#   - 주석산 농도
#   - 아세트산 농도
#   - 구연산 농도
#   - 잔류 당분 농도
#   - 염화나트륨 농도
#   - 유리아황산 농도
#   - 총 아황산 농도
#   - 밀도
#   - pH
#   - 황산칼륨 농도
#   - 알콜 도수
#   - 와인의 맛(0~10등급)
# 
# 
# * 클래스 : 1-레드와인, 0-화이트와인
#   

# In[2]:

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
import os
import tensorflow as tf


# In[3]:

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)


# In[4]:

df_pre  = pd.read_csv('data/wine.csv', header =None)
df = df_pre.sample(frac =1)

dataset = df.values
X = dataset[:, 0:12]
Y = dataset[:, 12]


# In[5]:

df


# In[63]:

df.info()


# In[64]:

model  = Sequential()
model.add(Dense(30, input_dim=12, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


# In[65]:

model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics =['accuracy'])


# In[68]:

MODEL_DIR = './data/model'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
MODEL_DIR


# In[79]:

# 모델 저장 조건 설정

modelpath='./data/model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath=modelpath, monitor = 'val_loss', 
                               verbose=1, save_best_only=True )

# verbose = 1 이면 해당 함수의 진행 사항이 출력
# verbose = 0 이면 출력되지 않는다.


# In[80]:

# 학습 자동 중단

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100)


# In[ ]:




# In[81]:

# 모델 실행 및 저장

history = model.fit(X, Y, 
                      validation_split=0.33, 
                      epochs=3500, 
                      batch_size=500,   ## 500개를 가지고 3500번을 돌리낟.
                      verbose=0, callbacks=[early_stopping_callback, checkpointer]);


# In[73]:

# y_vloss에 테스트 셋으로 실험 결과의 오차 값을 저장

y_vloss = history.history['val_loss']

# y_acc에 학습셋으로 실험 결과의 오차 값을 저장

y_acc = history.history['acc']


# In[74]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# x값을 지정하고 정확도를 파란색으로, 오차를 빨간색으로 표시

x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vloss, 'o', c='red' , markersize=3)
plt.plot(x_len, y_acc,   'o', c='blue', markersize=3)

plt.show()


# In[ ]:




# In[ ]:



