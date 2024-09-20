"""
Created on Fri Aug  5 10:28:27 2022

@authors: Sabyasachi Biswas, Cemre Ömer Ayna
"""
import numpy as np
from random import shuffle
import tensorflow as tf
import keras
from math import ceil
import matplotlib.pyplot as plt 
import math
import cmath
import h5py
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

print(tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

BATCH = 16
EPOCH = 50 

data = h5py.File("C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/Traffic_Sign/Radar+Camera/Dataset/all_data.h5", 'r')
x_train = np.asarray(data["xtrain_fusion"])
y_train = np.asarray(data["ytrain_fusion"])
x_test = np.asarray(data["xtest_fusion"])
y_test = np.asarray(data["ytest_fusion"])

input_data = tf.keras.Input(shape=x_train.shape[1:])

x = tf.keras.layers.Conv3D(12, (3,3,3))(input_data)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.activations.relu(x)
x = tf.keras.layers.MaxPooling3D(pool_size=2)(x)
#x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Conv3D(12, (3,3,3))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.activations.relu(x)
x = tf.keras.layers.MaxPooling3D(pool_size=2)(x)
#x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Conv3D(16, (3,3,3))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.activations.relu(x)
x = tf.keras.layers.MaxPooling3D(pool_size=2)(x)
#x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Conv3D(16, (3,3,3))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.activations.relu(x)
x = tf.keras.layers.MaxPooling3D(pool_size=2)(x)
#x = tf.keras.layers.Dropout(0.2)(x)

flatten = tf.keras.layers.Flatten()(x)
dense = tf.keras.layers.Dense(64)(flatten)
#dropout = tf.keras.layers.Dropout(0.5)(dense)
relu_dense = tf.keras.activations.relu(dense)

dense2 = tf.keras.layers.Dense(12)(dense)
output_layer = tf.keras.activations.softmax(dense2)

model = tf.keras.Model(inputs=input_data, outputs=output_layer)
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

history=model.fit(x=x_train,
        y=y_train,
        batch_size=BATCH,
        epochs=EPOCH,
        validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

y_pred = model.predict(x_test)

#plotting confusion_matrix#####################################################

pred_y = []
for i in range(len(y_pred)):
    list_y = list(y_pred[i])
    a = list_y.index(max(list_y))
    pred_y.append(a)
    
cf_matrix = confusion_matrix(y_test, pred_y)

print(cf_matrix)

import seaborn as sns

plt.figure(1)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.savefig('C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/Traffic_Sign/Radar+Camera/results/Fusion_confusion_matrix.png')
plt.show()

plt.figure(2)
cf = []
for i in range(len(cf_matrix)):
    cf.append(cf_matrix[i].astype(np.float64)/sum(cf_matrix[i].astype(np.float64)))
    
ax = sns.heatmap(cf, annot=True, cmap='Blues')

ax.set_title('Confusion Matrix in %\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.savefig('C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/Traffic_Sign/Radar+Camera/results/Fusion_confusion_matrix%.png')
plt.show()

# ##################################################

# train_acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# train_loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(len(train_acc))

# plt.plot(epochs, train_acc, 'r', label='Training acc',linewidth=1)
# plt.plot(epochs, val_acc, 'b', label='Validation acc',linewidth=1)
# plt.title('Training and Validation Accuracy',fontsize=14)
# plt.ylabel('Accuracy',fontsize=14) 
# plt.xlabel('Epoch',fontsize=14)
# plt.legend()
# plt.show()
# plt.savefig('C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/Traffic Sign/results/train_vs_val.png')

# plt.plot(epochs, train_loss, label='Training loss',linewidth=2)
# plt.plot(epochs, val_loss, label='validation Loss',linewidth=2)
# plt.title('Training and Validation Losses',fontsize=14)
# plt.ylabel('Loss',fontsize=14) 
# plt.xlabel('Epoch',fontsize=14)
# plt.legend()
# plt.show()
# plt.savefig('C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/Traffic Sign/results/trainloss_vs_valloss.png')
# #plt.ylim([0, 10])


# # model.save(".\\models\\examples\\sincnet1d")
# # model.save_weights('.\\models\\examples\\sincnet_weights.h5')
