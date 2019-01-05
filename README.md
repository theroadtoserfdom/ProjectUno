# ProjectUno
Windows 10 - Anaconda, Jupyter Notebook, Tensorflow, Keras 

# Course Objective:
This project aims to take you from a complete beginner to building your first deep learning application. 

# After completing this project you will be able to: 

1. Navigate virtual environments in Anaconda 
2. Enable environments in Jupyter Notebook 
3. Install dependencies for Tensorflow and Keras
4. Execute and Troubleshoot code with Tensorflow and Keras libraries

# Before we begin:
- Uninstall any other versions of Python installed on the Windows Machine
- Download the latest Anaconda Installer from https://www.anaconda.com/download/
  - Be sure to select the proper bit installer (E.g. x64 64-Bit or x86 32-Bit) 
  - I chose to NOT add Anaconda to my PATH environment variable (this gives Python access to our CMD terminal via the Python Command)
  
# Exclaimer: At the making of this tutorial Tensorflow only supports Python 3.6  
(I.e. no need to worry about the Python Version (ex. 3.6) this can be specified within our Virtual Environments)

# Once you have Anaconda Installed: Let's set up our Virtual Environment & Install our Dependencies

To Create Virtual Environments within Anaconda: 
- conda create -n ProjectUno python=3.6
- Proceed ([y]/n)? y - to Update Packages 

To List All Virtual Environments within Anaconda:
- conda env list 

To Enter a Virtual Environment 
- activate ProjectUno

To Check Python Dependency Versions
- python --version 
- jupyter notebook --version

To Install Python Dependencies
- conda install -c conda-forge tensorflow 
- pip install keras 
- pip install matplotlib

To Enable our Virtual Environment within Jupyter Notebook
- pip install ipykernel
- python -m ipkernel install --user --name=ProjectUno

Open Jupyter Notebook
- Jupyter Notebook
- Select New\ProjectUno

Check Tensorflow Version
- import tensorflow as tf
- tf.__ version__ (Ignore space after first __)
  
To Execute code
- Ctrl + Enter 

To Enter New Cell 
- Shift + Enter 

# Congratulations! You have now set up your Deep Learning Environment!

We will begin by importing the mnist dataset from keras (https://keras.io/datasets/)
- mnist = tf.keras.datasets.mnist  # 28x28 images of hand-written digits 0-9
- (x_train, y_train), (x_test, y_test) = mnist.load_data()

Our dataset can be examined using the matplotlib dependency
- import matplotlib.pyplot as plt
- print(x_train[0]) #(This function displays our tensor (feel free to change the digit to examine the data)) 

We will use matplotlib to display our image data (feel free to change the digit to examine additional images)
- plt.imshow(x_train[0], cmap = plt.cm.binary)  # cmap = color map , binary = black & white  
- plt.show()
- print(x_train[0])

We will begin to Normalize our dataset using Keras: Adding to our code from above 
- mnist = tf.keras.datasets.mnist  # 28x28 images of hand-written digits 0-9
- (x_train, y_train), (x_test, y_test) = mnist.load_data()
- x_train = tf.keras.utils.normalize(x_train, axis=1)
- x_test = tf.keras.utils.normalize(x_test, axis=1) # Be sure to Ctrl + Enter at every cell to commit code 

# Normalizing our data allows our pixel data to be displayed between 0-1 instead of 0-255. This allows our network to process our data much faster.

Below we will begin to build our model using keras 
- mnist = tf.keras.datasets.mnist  # 28x28 images of hand-written digits 0-9
- (x_train, y_train), (x_test, y_test) = mnist.load_data()
- x_train = tf.keras.utils.normalize(x_train, axis=1)
- x_test = tf.keras.utils.normalize(x_test, axis=1) # Be sure to Ctrl + Enter at every cell to commit code 
- model = tf.keras.models.Sequential()
- model.add(tf.keras.layers.Flatten())  # this is our import layer 
- model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # this is our hidden layer using rectified linear activation function
- model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # 128 is the number of neurons 
- model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # 10 is the number of classifications , Softmax = Probability 

Now that we have our Model Architecture Built, we will begin our Training Architecture 
- model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
- These are the most basic criteria for compiling a model

To Begin Training
- model.fit(x_train, y_train, epochs=3)  
- "An epoch is a measure of the number of times all of the training vectors are used once to update the weights."
- https://nl.mathworks.com/matlabcentral/answers/62668-what-is-epoch-in-neural-network
- Using Jupyter Notebook: Ctrl + Enter will execute our code 






















