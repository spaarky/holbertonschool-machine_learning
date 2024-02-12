#!/usr/bin/env python3
"""_summary_
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015)
    """
    init = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))

    # conv1 layer
    conv1 = K.layers.Conv2D(64, kernel_size=7, strides=2,
                            padding='same', kernel_initializer=init)(X)
    norm1 = K.layers.BatchNormalization()(conv1)
    act1 = K.layers.Activation('relu')(norm1)

    # max pool layer
    max_pool1 = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                      padding='same')(act1)

    # block 1 - conv2_x
    proj1 = projection_block(max_pool1, [64, 64, 256], s=1)
    iden1 = identity_block(proj1, [64, 64, 256])
    iden2 = identity_block(iden1, [64, 64, 256])

    # block 2 - conv3_x
    proj2 = projection_block(iden2, [128, 128, 512], s=2)
    iden3 = identity_block(proj2, [128, 128, 512])
    iden4 = identity_block(iden3, [128, 128, 512])
    iden5 = identity_block(iden4, [128, 128, 512])

    # block 3 - conv4_x
    proj3 = projection_block(iden5, [256, 256, 1024], s=2)
    iden6 = identity_block(proj3, [256, 256, 1024])
    iden7 = identity_block(iden6, [256, 256, 1024])
    iden8 = identity_block(iden7, [256, 256, 1024])
    iden9 = identity_block(iden8, [256, 256, 1024])
    iden10 = identity_block(iden9, [256, 256, 1024])

    # block 4 - conv5_x
    proj4 = projection_block(iden10, [512, 512, 2048], s=2)
    iden11 = identity_block(proj4, [512, 512, 2048])
    iden12 = identity_block(iden11, [512, 512, 2048])

    # average pool layer
    avg_pool = K.layers.AveragePooling2D(pool_size=7, strides=1)(iden12)

    # fully connected layer
    dense = K.layers.Dense(1000, activation='softmax',
                           kernel_initializer=init)(avg_pool)

    model = K.models.Model(inputs=X, outputs=dense)

    return model
