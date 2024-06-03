#!/usr/bin/env python3
"""Summary"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model) :

    def __init__( self, generator , discriminator , latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005):
        """Summary"""
        super().__init__()                         # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples    = real_examples
        self.generator        = generator
        self.discriminator    = discriminator
        self.batch_size       = batch_size
        self.disc_iter        = disc_iter

        self.learning_rate    = learning_rate
        self.beta_1=.5
        self.beta_2=.9

        # define the generator loss and optimizer:
        self.generator.loss      = lambda x : tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer , loss=generator.loss )

        # define the discriminator loss and optimizer:
        self.discriminator.loss      = lambda x,y : tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) + tf.keras.losses.MeanSquaredError()(y, -1*tf.ones(y.shape))
        self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer , loss=discriminator.loss )


    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """Summary"""
        if not size :
            size= self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        """Summary"""
        if not size :
            size= self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices  = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # overloading train_step()
    def train_step(self,useless_argument):
        """Summary"""
        for _ in range(self.disc_iter):
        # compute the loss for the discriminator in a tape watching the discriminator's weights
            # get a real sample
            real_sample = self.get_real_sample()
            # get a fake sample
            fake_sample = self.get_fake_sample()
                # compute the loss discr_loss of the discriminator on real and fake samples
            with tf.GradientTape() as tape:
                disc_real = self.discriminator(real_sample)
                disc_fake = self.discriminator(fake_sample)

                disc_loss = self.discriminator.loss(disc_real, disc_fake)
            # apply gradient descent once to the discriminator
            disc_gradient = tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.discriminator.optimizer.apply_gradients(zip(disc_gradient, self.discriminator.trainable_variables))

        # compute the loss for the generator in a tape watching the generator's weights
        with tf.GradientTape() as tape:
            # get a fake sample
            fake_sample = self.get_fake_sample()
            gen_output = self.discriminator(fake_sample, training=True)

            # compute the loss gen_loss of the generator on this sample
            gen_loss = self.generator.loss(gen_output)

        # apply gradient descent to the discriminator
        gen_gradient = tape.gradient(gen_loss, self.generator.trainable_variables)

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
