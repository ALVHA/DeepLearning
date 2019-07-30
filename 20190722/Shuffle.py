
# coding: utf-8

# In[9]:

import hello
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
## 매번 같은 데이터로 할 순 없으니 셔플한다.


# In[6]:

A = pd.DataFrame(np.random.rand(5,1)) ## 5행1열로 데이터를 바으라
A


# In[7]:

B =pd.DataFrame(np.random.rand(5,2))
B


# In[8]:

C = pd.DataFrame(np.random.rand(5,3))
C


# In[30]:

A_, B_, C_ = shuffle(A,B,C)


# In[31]:

print(A_, B_, C_)


# In[32]:

D = pd.DataFrame(np.random.rand(4,2))
D


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



