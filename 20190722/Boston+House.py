
# coding: utf-8

# # 보스톤 집 값 예측하기
# 
# * 보스턴 주택 데이터는 여러 개의 측정지표들을 포함한, 보스톤 인근의 주택가 중앙값
# 
# * Variable in order:
#   - CRIM : 마을별 1인당 범죄율
#   - ZN : 25,000 평방미터를 초과하는 거주지역의 비율
#   - INDUS : 비소매 상업지역이 점유하고 있는 토지의 비율
#   - CHAS : 찰스 강에 대한 더미변수(강의 경계에 위치한 경우는 1, 아니면 0
#   - NOX : 10ppm 당 농축 일산화질소
#   - RM  : 주책 1가구당 평균 방의 개수
#   - AGE : 1940년 이전에 건축된 소유 주택의 비율
#   - DIS : 5개의 보스톤 직업센터까지의 접근성 지수
#   - RAD : 방사형 돌까지의 접근성 지수
#   - TAX : 10,000달러 당 재산세율
#   - PTRATIO : 시별 학생/교사 비율
#   - B   : 1000(Bk - 0.63) ^2, 여기서 Bk는 시별 흑인의 비율
#   - LSTAT : 모집단의 하위계층의 비율(%)
#   - MEDV : 본인 소유의 주택가격(중앙값) (단위:$1,000)

# In[24]:

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf


# In[25]:

## 랜덤 시드값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)


# In[26]:

df =pd.read_csv('./data/housing.csv', delim_whitespace = True, header =None)

dataset = df.values
X = dataset[:, 0:13] 
Y = dataset[:, 13]  ## 집값이 지 ㅇ낳을까?


# In[33]:

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=seed)


# In[30]:

## 어떤 알고리즘을 적용할 것인지 선택
model = Sequential()
model.add(Dense(30, input_dim = 13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))


# In[31]:

model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(X_train, Y_train, epochs = 200, batch_size =10)


# In[34]:

# 예측값과 실제 값의 비교

Y_prediction = model.predict(X_test).flatten()
for i in range(10):
    label = Y_test[i]
    prediction = Y_prediction[i]
    print('실제가격 : {:.3f}, 예상가격 : {:.3f}'.format(label, prediction))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



