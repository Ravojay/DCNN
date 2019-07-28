from functools import wraps
import numpy as np
from keras import backend as K
from keras.layers import concatenate, Activation, Conv1D, Conv2D, Softmax, Multiply, Add, Flatten, Dropout, Dense, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.models import Model
from utils import compose
from keras.backend import squeeze
from keras.layers.convolutional import ZeroPadding1D
from keras.layers.pooling import AveragePooling1D
from keras import regularizers

featureSize = 128
denseLayerNb = 5#at least one

@wraps(Embedding)
def denseAttentionEbd(*args, **kwargs):
    '''Wrapper to set daNet parameters for Embedding layer'''
    daEbdKwargs = {'mask_zero' : False}
    daEbdKwargs['trainable'] = False
    daEbdKwargs.update(kwargs)
    return Embedding(*args, **daEbdKwargs)

def bnAndRelu():
    return compose(
        BatchNormalization(),
        Activation('relu'))

def convertFeature(*args, **kwargs):
    return translationConv1D(*args, **kwargs)

@wraps(Conv1D)
def newConv1D(*args, **kwargs):
    '''Wrapper to set kernel_regularizer to Conv1D layer'''
    ncKwargs = {}#'kernel_regularizer' : regularizers.l2(0.001)}
    ncKwargs.update(kwargs)
    return Conv1D(*args, **ncKwargs)

@wraps(Dense)
def newDense(*args, **kwargs):
    '''Wrapper to set the kernel_regularizer to Dense layer'''
    ndKwargs = {}#'kernel_regularizer' : regularizers.l1(0.001)}
    ndKwargs.update(kwargs)
    return Dense(*args, **ndKwargs)

@wraps(newConv1D)
def translationConv1D(*args, **kwargs):
    '''Wrapper to set the tranNet parameters for Conv1D layer'''
    cfKwargs = {'kernel_size' : 1}
    cfKwargs['filters'] = featureSize
    cfKwargs['strides'] = 1
    cfKwargs['padding'] = 'valid'
    cfKwargs.update(kwargs)
    return newConv1D(*args, **cfKwargs)

@wraps(newConv1D)
def denseConv1D(*args, **kwargs):
    '''Wrapper to set the denseNet parameters for Conv1D layer'''
    dcKwargs = {'kernel_size' : 3}
    dcKwargs['filters'] = featureSize
    dcKwargs['strides'] = 1
    dcKwargs['padding'] = 'valid'
    #dcKwargs['kernel_regularizer'] = regularizers.l2(0.001)
    dcKwargs.update(kwargs)
    return compose(ZeroPadding1D(padding=1),
                   newConv1D(*args, **dcKwargs))

@wraps(AveragePooling1D)
def denseAvgPool(*args, **kwargs):
    '''Wrapper to set the densePool parameters for AvgPool layer'''
    dapKwargs = {'pool_size' : 3}
    dapKwargs['strides'] = 2
    dapKwargs['padding'] = 'valid'
    dapKwargs.update(kwargs)
    return compose(ZeroPadding1D(padding=1),
                   AveragePooling1D(*args, **dapKwargs))

@wraps(Conv2D)
def groupConv1D(*args, **kwargs):
    '''Wrapper to set the group convolution parameters for Conv2D layer'''
    gcKwargs = {'kernel_size' : (1, 1)}
    gcKwargs['filters'] = featureSize
    gcKwargs['strides'] = 1
    gcKwargs['padding'] = 'valid'
    #gcKwargs['kernel_regularizer'] = regularizers.l2(0.001)
    gcKwargs.update(kwargs)
    return Conv2D(*args, **gcKwargs)

@wraps(newDense)
def attentionFunc1(*args, **kwargs):
    '''Wrapper to set the attention function1 parameters for Dense layer'''
    af1Kwargs = {'units' : 64}
    af1Kwargs.update(kwargs)
    return compose(newDense(*args, **af1Kwargs),
                   Activation('relu'))

@wraps(newDense)
def attentionFunc2(*args, **kwargs):
    '''Wrapper to set the attention function2 parameters for Dense layer'''
    af2Kwargs = {'units' : 32}
    af2Kwargs.update(kwargs)
    return compose(newDense(*args, **af2Kwargs),
                   Activation('relu'))

@wraps(newDense)
def attentionFunc3(*args, **kwargs):
    '''Wrapper to set the attention function3 parameters for Dense layer'''
    af3Kwargs = {'units' : 4 + denseLayerNb - 1}
    af3Kwargs.update(kwargs)
    return compose(newDense(*args, **af3Kwargs),
                   Activation('relu'))

@wraps(newDense)
def finalFunc1(*args, **kwargs):
    '''Wrapper to set the final function1 parameters for Dense layer'''
    fc1Kwargs = {'units' : 64}
    #fc1Kwargs['kernel_regularizer'] = regularizers.l2(0.001)
    fc1Kwargs.update(kwargs)
    return compose(Dropout(rate=0.7),
                   newDense(*args, **fc1Kwargs),
                   Activation('relu'))

@wraps(newDense)
def finalFunc2(*args, **kwargs):
    '''Wrapper to set the final function2 parameters for Dense layer'''
    fc2Kwargs = {}#{'kernel_regularizer' : regularizers.l2(0.001)}
    fc2Kwargs.update(kwargs)
    return compose(Dropout(rate=0.7),
                   Dense(*args, **fc2Kwargs),
                   Activation('relu'))

def translationAndConcat(inputs):
    outputs1 = convertFeature()(inputs)
    outputs = bnAndRelu()(outputs1)
    outputs2 = translationConv1D()(outputs)
    outputs = bnAndRelu()(outputs2)
    outputs3 = translationConv1D()(outputs)
    outputs = bnAndRelu()(outputs3)
    outputs4 = denseConv1D()(outputs)
    outputs = concatenate([outputs4, outputs1, outputs2, outputs3])
    outputs = bnAndRelu()(outputs)
    return outputs

def denseBody(inputs):
    outputs = inputs
    for _ in range(denseLayerNb - 1):
        outputs1 = denseAvgPool()(outputs)
        outputs2 = denseConv1D()(outputs1)
        outputs = concatenate([outputs2, outputs1])
        outputs = bnAndRelu()(outputs)
    return outputs

def slice(x, h1, h2, w1, w2):
    return x[:, h1 : h2, w1 : w2, :]

def sliceChannel(x, c1, c2):
    return x[:, :, c1 : c2]

def sliceWords(x, w1, w2):
    return x[:, w1 : w2, :]

def attentionPre(inputs, wordsLength):
    #remember to use layer rather than tensor in a Model all the time
    outputs = inputs
    for _ in range(2):
        #group convolution begin
        outputs = Reshape((wordsLength, 4 + denseLayerNb - 1, featureSize))(outputs)
        L = [Lambda(slice, arguments={'h1':0, 'h2':wordsLength, 'w1':i, 'w2':i + 1})(outputs) for i in range(4 + denseLayerNb - 1)]
        L = [groupConv1D()(e) for e in L]
        L = [Lambda(lambda z : squeeze(z, - 2))(e) for e in L]
        outputs = concatenate(L)
        #group convolution end
        outputs = bnAndRelu()(outputs)
    return outputs

def sliceWordsDenseConcatWords(inputs, wordsLength):
    L = [Lambda(sliceWords, arguments={'w1': i, 'w2': i + 1})(inputs) for i in range(wordsLength)]
    L = [attentionFunc1()(e) for e in L]
    L = [attentionFunc2()(e) for e in L]
    L = [attentionFunc3()(e) for e in L]
    L = [Reshape((1, 4 + denseLayerNb - 1))(e) for e in L]
    outputs = concatenate(L, axis=-2)
    return outputs

def generateAttentions(inputs):
    outputsList = [Lambda(sliceChannel, arguments={'c1': i, 'c2': i + 1})(inputs) for i in range(4 + denseLayerNb - 1)]
    outputsList = [Lambda(K.tile, arguments={'n' : (1, 1, featureSize)})(e) for e in outputsList]
    return outputsList

def linearWithDropout(inputs, classNb):
    outputs = Flatten()(inputs)
    outputs = compose(finalFunc1(),
                      finalFunc2(units=classNb))(outputs)
    return outputs

def softmaxWithPreprocess(inputs):#to eliminate all the negative values
    outputs = compose(Activation('relu'),
                   Softmax())(inputs)
    return outputs

def modelBody(inputs, ebdMatrix, dataWords, classNb):
    finalWordsLength = dataWords
    layersNb = denseLayerNb - 1
    pad = 1
    winSize = 3
    stride = 2
    for _ in range(layersNb):
        finalWordsLength = (finalWordsLength + pad * 2 - winSize) // stride + 1
    outputs = denseAttentionEbd(weights=[ebdMatrix], input_length=dataWords, input_dim=np.shape(ebdMatrix)[0], output_dim=np.shape(ebdMatrix)[1])(inputs)
    outputs = translationAndConcat(outputs)
    outputs = denseBody(outputs)
    outputs = attentionPre(outputs, finalWordsLength)
    #get x here begin
    xList = [Lambda(sliceChannel, arguments={'c1': i * featureSize, 'c2': (i + 1) * featureSize})(outputs) for i in range(4 + denseLayerNb - 1)]
    #get x here end
    #reduction and concat begin
    outputsList = [Lambda(K.sum, arguments={'axis':-1, 'keepdims':True})(e) for e in xList]
    outputs = concatenate(outputsList)
    #reduction and concat end
    outputs = sliceWordsDenseConcatWords(outputs, finalWordsLength)
    outputs = softmaxWithPreprocess(outputs)#softmax for axis=-1 that is for each attention implement softmax
    aList = generateAttentions(outputs)#get a here
    reweightedList = [Multiply()([x, a]) for x, a in zip(xList, aList)]
    outputs = Add()(reweightedList)
    outputs = linearWithDropout(outputs, classNb)
    outputs = softmaxWithPreprocess(outputs)
    return Model(inputs, outputs)
