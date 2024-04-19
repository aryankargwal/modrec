import os
import numpy as np
import keras.models as models
from keras.layers import Reshape,Dense,Dropout,Activation,Flatten, Input
from keras.layers import Convolution2D,Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation
from keras.models import Model

# Build VT-CNN2 Neural Net model using Keras primitives --
#  - Reshape [N,2,128] to [N,2,128,1] on input
#  - Pass through 2 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross entropy optimization

def CNN2Model(weights=None,
             input_shape=[2,128],
             classes=11,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5 # dropout rate (%)

    input = Input(shape=input_shape + [1], name='input')


    x = Reshape(target_shape=input_shape + [1])(input)


    x = Conv2D(50, (1, 8), padding='same', activation="relu", name="conv1", kernel_initializer='glorot_uniform')(x)
    x = Dropout(dr)(x)

    x = Conv2D(50, (2, 8), padding="valid", activation="relu", name="conv2", kernel_initializer='glorot_uniform')(x)
    x = Dropout(dr)(x)

    x = Flatten()(x)

    x = Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1")(x)
    x = Dropout(dr)(x)

    x = Dense(classes, kernel_initializer='he_normal', name="dense2")(x)
    output = Activation('softmax')(x)

    model = Model(inputs=input, outputs=output)
    
    '''
    #Qkeras version

    q_bit = quantized_bits(16, 8)

    model.add(QConv2D(50, (1, 8), padding='same', activation="quantized_relu(16, 8)", name="conv1", kernel_initializer='glorot_uniform', 
              kernel_quantizer= q_bit, bias_quantizer=q_bit))

    model.add(Dropout(dr))

    model.add(QConv2D(50, (2, 8), padding="valid", activation="quantized_relu(16, 8)", name="conv2", kernel_initializer='glorot_uniform',
              kernel_quantizer= q_bit, bias_quantizer=q_bit))

    model.add(Dropout(dr))

    model.add(Flatten())

    model.add(QDense(256, activation='quantized_relu(16, 8)', kernel_initializer='he_normal', name="dense1",
              kernel_quantizer= q_bit, bias_quantizer=q_bit))

    model.add(Dropout(dr))

    model.add(QDense(classes, kernel_initializer='he_normal', name="dense2",
              kernel_quantizer= q_bit, bias_quantizer=q_bit))

    model.add(Activation('softmax'))
    '''
    
    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

if __name__ == '__main__':
    print(CNN2Model().summary())
