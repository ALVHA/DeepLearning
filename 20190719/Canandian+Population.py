
# coding: utf-8

# In[17]:

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

## year, population


# In[25]:

df = pd.read_csv('./data/CanPop.csv')


# In[41]:

df


# In[71]:

a = df['year']
b = df['population']
b = b.replace('.', ' ')


# In[77]:

plt.figure(figsize = (10,6))
plt.plot(a, b)

plt.xlabel('year')
plt.ylabel('Population')
plt.title('Canadian Population Change, Mil')
plt.show()


# In[ ]:




# In[ ]:



