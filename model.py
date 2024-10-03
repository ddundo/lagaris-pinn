import tensorflow as tf
import numpy as np


class Model():
    def __init__(self, 
                 pde: callable,
                 net: tf.keras.Model,
                 optimiser: tf.keras.optimizers.Optimizer,
                 ):
        """
        Base class for neural network models.

        Args:
            pde (callable): The loss function with partial differential equation to solve.
            net (tf.keras.Model): The neural network model.
            optimiser (tf.keras.optimizers.Optimizer): The optimiser to use for training.
        """
        self.pde = pde
        self.net = net
        self.optimiser = optimiser

    def __call__(self, x: tf.Tensor):
        """
        Call the neural network model.

        Args:
            x (tf.Tensor): The input tensor.

        Returns:
            The output of the neural network model.
        """
        return self.net(x)

    def train(self, 
              x: tf.Tensor, 
              num_epochs: int=10000, 
              verbose: bool = False,
              ):
        """
        Train the model.

        Args:
            x (tf.Tensor): The training input tensor.
            num_epochs (int, optional): The number of epochs to train for. Defaults to 10000.
            verbose (bool, optional): Whether to print training progress. Defaults to False.

        Returns:
            losses (np.array): The training losses for each epoch.
        """
        
        @tf.function
        def train_step(x):
            with tf.GradientTape(persistent=True) as tape:
                loss = self._loss(x, self.net)
            grads = tape.gradient(loss, self.net.trainable_variables)
            self.optimiser.apply_gradients(zip(grads, self.net.trainable_variables))
            return loss
        
        losses = np.zeros(num_epochs)
        for epoch in range(num_epochs):
            losses[epoch] = train_step(x)
            if (verbose and (epoch+1) % 1000 == 0) or (epoch == num_epochs-1):
                print(f'Epoch {epoch+1} -- Training loss: {losses[epoch]}')
        return losses
    
    def test(self, x: tf.Tensor):
        """
        Evaluate the model.

        Args:
            x (tf.Tensor): The input tensor.

        Returns:
            The loss for the input tensor.
        """
        return self._loss(x, self.net)
    
    def _loss(self, x, nn):
        return self.pde(x, nn)
