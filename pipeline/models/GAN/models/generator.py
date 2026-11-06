"""
Multi-conditional Generator for scGAN
Uses conditional batch normalization with multiple mixed conditions
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ConditionalBatchNorm(layers.Layer):
    """
    Conditional Batch Normalization layer that modulates based on conditions
    """

    def __init__(self, units, condition_dim, **kwargs):
        super(ConditionalBatchNorm, self).__init__(**kwargs)
        self.units = units
        self.condition_dim = condition_dim

        # Batch normalization layer
        self.batch_norm = layers.BatchNormalization()

        # MLP to generate scale (gamma) and offset (beta) from conditions
        self.condition_mlp = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(units * 2)  # Output both gamma and beta
        ])

    def call(self, inputs, conditions, training=None):
        # Apply standard batch normalization
        normalized = self.batch_norm(inputs, training=training)

        # Generate scale and offset from conditions
        cond_params = self.condition_mlp(conditions)
        gamma = cond_params[:, :self.units]
        beta = cond_params[:, self.units:]

        # Apply conditional modulation
        # gamma * normalized + beta
        output = normalized * (1 + gamma) + beta

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'condition_dim': self.condition_dim
        })
        return config


class MultiConditionalGenerator(keras.Model):
    """
    Generator with multi-conditional batch normalization

    Architecture:
    Input: noise (latent_dim) + conditions (condition_dim)
    Hidden layers: [256, 512, 1024] with conditional batch norm
    Output: 2000 genes (normalized expression values)
    """

    def __init__(self,
                 latent_dim=128,
                 condition_dim=6,  # Will be set by condition processor
                 output_dim=2000,
                 hidden_layers=None,
                 **kwargs):
        super(MultiConditionalGenerator, self).__init__(**kwargs)

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers or [256, 512, 1024]

        # Build the network layers
        self.dense_layers = []
        self.cond_bn_layers = []

        for i, units in enumerate(self.hidden_layers):
            self.dense_layers.append(
                layers.Dense(units, name=f'gen_dense_{i}')
            )
            self.cond_bn_layers.append(
                ConditionalBatchNorm(units, condition_dim, name=f'gen_cbn_{i}')
            )

        # Output layer (no batch norm, just dense + activation)
        self.output_layer = layers.Dense(output_dim, activation=None, name='gen_output')

    def call(self, inputs, training=None):
        """
        Forward pass

        Args:
            inputs: Tuple of (noise, conditions)
                - noise: (batch_size, latent_dim)
                - conditions: (batch_size, condition_dim)
            training: Whether in training mode

        Returns:
            Generated expression values (batch_size, output_dim)
        """
        noise, conditions = inputs

        # Concatenate noise and conditions for initial input
        x = tf.concat([noise, conditions], axis=1)

        # Pass through hidden layers with conditional batch norm
        for dense_layer, cbn_layer in zip(self.dense_layers, self.cond_bn_layers):
            x = dense_layer(x)
            x = cbn_layer(x, conditions, training=training)
            x = tf.nn.relu(x)

        # Output layer - use ReLU to ensure non-negative values
        # (since we're generating normalized log-expression data)
        output = self.output_layer(x)
        output = tf.nn.relu(output)

        return output

    def generate(self, conditions, num_samples=None):
        """
        Generate samples given conditions

        Args:
            conditions: Condition tensor (batch_size, condition_dim)
            num_samples: If provided, will use this instead of conditions batch size

        Returns:
            Generated expression values
        """
        if num_samples is not None:
            batch_size = num_samples
        else:
            batch_size = tf.shape(conditions)[0]

        # Sample noise from standard normal
        noise = tf.random.normal((batch_size, self.latent_dim))

        # Generate samples
        return self.call((noise, conditions), training=False)

    def get_config(self):
        return {
            'latent_dim': self.latent_dim,
            'condition_dim': self.condition_dim,
            'output_dim': self.output_dim,
            'hidden_layers': self.hidden_layers
        }


def build_generator(latent_dim=128, condition_dim=6, output_dim=2000,
                    hidden_layers=None):
    """
    Build and return a multi-conditional generator

    Args:
        latent_dim: Dimension of noise input
        condition_dim: Dimension of condition input
        output_dim: Number of genes to generate
        hidden_layers: List of hidden layer sizes

    Returns:
        MultiConditionalGenerator model
    """
    generator = MultiConditionalGenerator(
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        output_dim=output_dim,
        hidden_layers=hidden_layers
    )

    # Build the model by calling it once
    dummy_noise = tf.zeros((1, latent_dim))
    dummy_conditions = tf.zeros((1, condition_dim))
    _ = generator((dummy_noise, dummy_conditions))

    return generator