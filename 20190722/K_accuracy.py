
# coding: utf-8

# In[2]:

from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np
import tensorflow as tf


# In[3]:

# seed 값 설정

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)


# In[4]:

# 데이터 입력

df = pd.read_csv('./data/sonar.csv', header=None)
print(df.info())


# In[5]:

dataset = df.values
X = dataset[:, 0:60]   # 속성
Y_obj = dataset[:,60]  # 클래스


# In[6]:

# 문자열 변환:   문자 --> 숫자 --> 이진코드

e = LabelEncoder()  
e.fit(Y_obj) ## ---> 숫자
Y = e.transform(Y_obj) ##  --> 이진코드
print(Y)


# In[7]:

# 10개의 파일로 쪼갬

n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)


# In[8]:

# 빈 accuracy 배열

accuracy = []


# In[9]:

# 모델의 설정, 컴파일, 실행

for train, test in skf.split(X,Y):
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1,  activation='sigmoid'))
    model.compile(loss='mean_squared_error',
                  optimizer = 'adam',
                  metrics=['accuracy'])
    model.fit(X[train], Y[train], epochs=100, batch_size=5)
    
    k_accuracy = "%.4f" % (model.evaluate(X[test], Y[test])[1])
    accuracy.append(k_accuracy)
    
# 결과 출력

print('\n %.f fold accuracy : ' % n_fold, accuracy)


# In[ ]:




# In[ ]:



