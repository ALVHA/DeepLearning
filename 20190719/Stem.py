
# coding: utf-8

# In[99]:

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


# In[100]:

df = pd.read_csv('./data/cane.csv')


# In[101]:

df.head(10)


# In[102]:

df.describe()


# ## 1. 표본을 추출했을 때 발병이 날 확률(평균) 완료
# 
# ## 2. 블록당 발병률
# 
# ## 3. 뿌리와 발병률과의 관계(비례)
# 

# In[103]:

## 표본을 추출했을 때 발병이 날 확률
# 발병률 = C/B * 100

df['DisRate'] = ( df['Disease']/df['Count'] ) * 100
df.head()


# In[106]:

df


# In[107]:

df.to_csv('./data/cane2.csv')


# In[108]:

df


# In[109]:

mean = df.DisRate[180]  ## 표본 하나당 발병이 날 확률
mean = int(mean)
mean


# In[110]:

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)


# In[172]:

ax = plt.subplots()
ax = sns.countplot('li', data = df, ax=ax[1])
plt.show()


# In[ ]:



