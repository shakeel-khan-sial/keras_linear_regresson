#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Model using Keras

# In[27]:


# imports
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt


# In[28]:


# data generation
x = data = np.linspace(1,5,1200)
y = x*4 + np.random.rand(*x.shape)* 0.3


# In[33]:


y


# # Custom Model Building with one layer

# In[9]:


model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))


# In[10]:


model.summary()


# ## Training and Testing

# In[11]:


model.compile(optimizer='sgd', loss='mse', metrics = ['mse'])


# In[12]:


weights = model.layers[0].get_weights()


# In[13]:


w_init = weights[0][0][0]


# In[14]:


b_init = weights[1][0]


# In[17]:


print('linear regression model is initialized with weights w: %.2f, b: %.2f' % (w_init, b_init))


# In[18]:


model.fit(x,y,batch_size = 1, epochs = 100, shuffle=False)


# In[19]:


weights = model.layers[0].get_weights()


# In[20]:


w_final = weights[0][0][0]


# In[21]:


b_final = weights[1][0]


# In[22]:


print('Linear regression model is trained to have weights w: %.2f, b:%.2f' % (w_final, b_final))


# In[23]:


predict = model.predict(data)


# ### Prediction

# In[26]:


plt.plot(data, predict,'b', data, y, 'k.')
plt.show()


# In[25]:





# In[ ]:




