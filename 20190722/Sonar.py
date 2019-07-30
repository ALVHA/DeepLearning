
# coding: utf-8

# In[1]:

from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


import pandas as pd
import numpy as np
import tensorflow as tf


# In[2]:

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)


# In[3]:

## 데이터 입력

df = pd.read_csv('./data/sonar.csv', header = None)
print(df.info())


# In[5]:

dataset = df.values
X =dataset[:, 0:60]
Y_obj = dataset[:,60]


# In[6]:

# 문자열 변환

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
print(Y)


# In[7]:

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = seed)


# In[8]:

# 모델 설정

model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1,  activation='sigmoid'))


# In[9]:

# 모델 컴파일

model.compile(loss = 'mean_squared_error',
              optimizer = 'adam',
              metrics = ['accuracy'])


# In[10]:

model.fit(X, Y, epochs = 130, batch_size =5)

model.save('./model/my_model_sonar.h5')


# In[ ]:

# 테스트를 위해 메모리 내의 모델을 삭제

del model

model = load_model('./model/my_model_sonar.h5')


# In[12]:

# 새롭게 불러온 모델로 테스트 실행

print('\n Test Accuracy : %.4f' % (model.evaluate(X_test, Y_test)[1]))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



