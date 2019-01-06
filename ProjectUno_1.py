#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf

tf.__version__


# In[4]:


tf.__version__


# In[5]:


import tensorflow as tf 

tf.__version__


# In[31]:


mnist = tf.keras.datasets.mnist  # 28x28 images of hand-written digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)


# In[33]:


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)


# In[34]:


model.save('ProjectUno.model')


# In[35]:


new_model = tf.keras.models.load_model('ProjectUno.model')


# In[36]:


predictions = new_model.predict([x_test])


# In[37]:


print(predictions)


# In[39]:


import numpy as np

print(np.argmax(predictions[0]))


# In[41]:


plt.imshow(x_test[0])
plt.show()


# In[32]:


import matplotlib.pyplot as plt 

plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
print(x_train[0])


# In[ ]:




