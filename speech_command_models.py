

from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, SeparableConv2D

    
def conv_network(input_shape=(40,101,1),
                depth_factor=1,
                num_F=12,
                dropout_prob=0.1,
                nclass=12,
                pool_type='avg',
                act='softmax'):
    
    net = Sequential()

    if pool_type == 'avg':
    	Pool = AveragePooling2D
    elif pool_type == 'max':
    	Pool = MaxPooling2D
    else:
    	raise ValueError('pool_type must be "avg" or "max"') 

        
    net.add(Conv2D(num_F, (3, 3), padding='same', input_shape=input_shape))
    net.add(BatchNormalization())
    net.add(Activation('relu'))

    for i in range(depth_factor-1):
        net.add(Conv2D(num_F, (3, 3), padding='same'))
        net.add(BatchNormalization())
        net.add(Activation('relu'))

    net.add(Pool(pool_size=(2, 2), padding='same'))
    
    for i in range(depth_factor):
        net.add(Conv2D(2*num_F, (3, 3), padding='same'))
        net.add(BatchNormalization())
        net.add(Activation('relu'))

    net.add(Pool(pool_size=(2, 2), padding='same'))

    for i in range(depth_factor):
        net.add(Conv2D(num_F*4, (3, 3), padding='same'))
        net.add(BatchNormalization())
        net.add(Activation('relu'))

    net.add(Pool(pool_size=(2, 2), padding='same'))

    for i in range(depth_factor):
        net.add(Conv2D(num_F*4, (3, 3), padding='same'))
        net.add(BatchNormalization())
        net.add(Activation('relu'))
   
    net.add(Pool(pool_size=(1, 13)))
 
    net.add(Dropout(dropout_prob))
    
    net.add(Flatten())
    net.add(Dense(nclass))
    net.add(Activation(act)) 

    return net


def separable_conv_network(input_shape=(40,101,1), dropout_prob=0.1, num_F=32, nclass=12):

    net = Sequential()

    net.add(Conv2D(num_F, (3, 3), padding='same', input_shape=input_shape))
    net.add(BatchNormalization())
    net.add(Activation('relu'))

    net.add(MaxPooling2D(pool_size=(2, 2)))

    net.add(SeparableConv2D(num_F*2, (3, 3), padding='same'))
    net.add(BatchNormalization())
    net.add(Activation('relu'))

    net.add(MaxPooling2D(pool_size=(2, 2)))

    net.add(Dropout(dropout_prob))
    net.add(SeparableConv2D(num_F*4, (3, 3), padding='same'))
    net.add(BatchNormalization())
    net.add(Activation('relu'))

    net.add(Dropout(dropout_prob))
    net.add(SeparableConv2D(num_F*4, (3, 3), padding='same'))
    net.add(BatchNormalization())
    net.add(Activation('relu'))

    net.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    net.add(Dropout(dropout_prob))
    net.add(SeparableConv2D(num_F*8, (3, 3), padding='same'))
    net.add(BatchNormalization())
    net.add(Activation('relu'))

    net.add(Dropout(dropout_prob))
    net.add(SeparableConv2D(num_F*8, (3, 3), padding='same'))
    net.add(BatchNormalization())
    net.add(Activation('relu'))

    net.add(AveragePooling2D(pool_size=(1, 13)))

    net.add(Flatten())
    net.add(Dense(nclass, activation = 'softmax')) 

    return net
