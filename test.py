from model import modelBody
import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.utils.np_utils import to_categorical
from keras.optimizers import Nadam, SGD
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.utils import plot_model

#import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#random.seed(2019)
#np.random.seed(2019)

#dataBatch = 8 * 10
#ebdFeatureNb = 300
#ebdWordsNb = 500
classNb = 4#4 for ag; 5 for yelp
dataWords = 128
batchSize = 256#256 for ag; 64 for yelp
dataDir = 'agnews/'#agnews for ag; yelp for yelp
weight_path = 'weight_ag_news/'#weight_ag_news for ag; weight_yelp for yelp

ebd = np.loadtxt(dataDir + 'local_dict.txt')
print(np.shape(ebd))

trainData = np.loadtxt(dataDir + 'train_data.txt')
print(np.shape(trainData))

testData = np.loadtxt(dataDir + 'test_data.txt')
print(np.shape(testData))

trainLabel = to_categorical(np.loadtxt(dataDir + 'train_label.txt')[1:], classNb)
print(np.shape(trainLabel))

testLabel = to_categorical(np.loadtxt(dataDir + 'test_label.txt')[1:], classNb)
print(np.shape(testLabel))
'''
#generate embedding or load embedding
#ebd : row = words, col = feature
ebd = np.random.rand(ebdWordsNb, ebdFeatureNb)
kebd = K.variable(value=ebd, dtype='float32', name='ebdMatrix')

#generate data or load data
#data : axis0 = batch, axis1 = words
data = np.zeros((dataBatch, dataWords))
for i in range(dataBatch):#each batch
    wordsLength = np.random.randint(dataWords // 2, dataWords + 1)
    for j in range(wordsLength):#words need to filled
        data[i, j] = np.random.randint(0, ebdWordsNb)
kdata = K.variable(value=data, dtype='int32', name='data')

#generate label or load label
#label : axis0 = classIndex from 0 to classNb - 1
label = np.random.randint(0, classNb, size=dataBatch)
label = to_categorical(label, classNb)
klabel = K.variable(value=label, dtype='float32', name='label')
'''
'''
print(ebd)
print(data)
print(label)
print(np.shape(ebd))
print(np.shape(data))
print(np.shape(label))
'''

def getBatches(data, batchSize):
    return (len(data) + batchSize - 1) // batchSize

def dataGenerator(data, targets, batchSize):
    batches = getBatches(data, batchSize)
    while (True):
        for i in range(batches):
            X = data[i * batchSize : (i + 1) * batchSize]
            Y = targets[i * batchSize : (i + 1) * batchSize]
            yield(X, Y)

K.clear_session()
gpu_options = tf.GPUOptions(allow_growth = True)
KTF.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count={'gpu':0})))
#the shape of Input is not including batch size
inputs = Input(shape=(dataWords, ))
md = modelBody(inputs, ebd, dataWords, classNb)
#plot_model(md, to_file='model.jpg', show_shapes=True)

opt = Nadam(lr=0.001, schedule_decay=0.0001)
#opt = SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False)
md.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['mae', 'acc'])
file_path = weight_path + "100ws-adam-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoints = ModelCheckpoint(file_path,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
callback_list = [checkpoints]
history = md.fit_generator(generator=dataGenerator(trainData, trainLabel, batchSize), epochs=20, steps_per_epoch=getBatches(trainData, batchSize), verbose=1, initial_epoch=0, validation_data=(testData, testLabel), validation_steps=getBatches(testData, batchSize), shuffle='batch', callbacks=callback_list)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('ag_accuracy_adam_100ws.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('ag_loss_adam_100ws.png')
plt.show()
