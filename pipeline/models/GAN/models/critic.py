"""
Multi-conditional Critic (Discriminator) for scGAN
Uses projection method for conditioning
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ConditionalProjection(layers.Layer):
    """
    Conditional projection layer that generates projection weights from conditions
    """

    def __init__(self, hidden_dim, condition_dim, **kwargs):
        super(ConditionalProjection, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        # MLP to generate projection weights from conditions
        self.projection_mlp = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(hidden_dim)  # Projection weights
        ])

    def call(self, hidden_features, conditions):
        """
        Apply conditional projection

        Args:
            hidden_features: Hidden layer features (batch_size, hidden_dim)
            conditions: Condition vector (batch_size, condition_dim)

        Returns:
            Projected value (batch_size, 1)
        """
        # Generate projection weights from conditions
        proj_weights = self.projection_mlp(conditions)  # (batch_size, hidden_dim)

        # Compute inner product between hidden features and projection weights
        # This is element-wise multiply then sum
        projection = tf.reduce_sum(hidden_features * proj_weights, axis=1, keepdims=True)

        return projection

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'condition_dim': self.condition_dim
        })
        return config


class MultiConditionalCritic(keras.Model):
    """
    Critic/Discriminator with multi-conditional projection

    Architecture:
    Input: expression (2000 genes)
    Hidden layers: [1024, 512, 256]
    Projection: condition-dependent projection
    Output: Wasserstein distance (scalar)
    """

    def __init__(self,
                 input_dim=2000,
                 condition_dim=6,
                 hidden_layers=None,
                 **kwargs):
        super(MultiConditionalCritic, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_layers = hidden_layers or [1024, 512, 256]

        # Build the network layers
        self.dense_layers = []
        for i, units in enumerate(self.hidden_layers):
            self.dense_layers.append(
                layers.Dense(units, activation='relu', name=f'critic_dense_{i}')
            )

        # Final hidden layer (no activation, will be used for projection)
        self.final_hidden = layers.Dense(
            self.hidden_layers[-1],
            activation=None,
            name='critic_final_hidden'
        )

        # Conditional projection layer
        self.cond_projection = ConditionalProjection(
            self.hidden_layers[-1],
            condition_dim,
            name='critic_projection'
        )

        # Final output layer (unconditional)
        self.output_layer = layers.Dense(1, activation=None, name='critic_output')

    def call(self, inputs, training=None):
        """
        Forward pass

        Args:
            inputs: Tuple of (expression, conditions)
                - expression: (batch_size, input_dim) - gene expression values
                - conditions: (batch_size, condition_dim) - encoded conditions
            training: Whether in training mode

        Returns:
            Critic score (batch_size, 1)
        """
        expression, conditions = inputs

        # Pass expression through hidden layers
        x = expression
        for dense_layer in self.dense_layers[:-1]:  # All but last layer
            x = dense_layer(x)

        # Get final hidden representation
        hidden = self.dense_layers[-1](x)

        # Apply conditional projection
        cond_score = self.cond_projection(hidden, conditions)

        # Get unconditional score
        uncond_score = self.output_layer(hidden)

        # Combine scores
        output = uncond_score + cond_score

        return output

    def get_config(self):
        return {
            'input_dim': self.input_dim,
            'condition_dim': self.condition_dim,
            'hidden_layers': self.hidden_layers
        }


def build_critic(input_dim=2000, condition_dim=6, hidden_layers=None):
    """
    Build and return a multi-conditional critic

    Args:
        input_dim: Dimension of expression input (number of genes)
        condition_dim: Dimension of condition input
        hidden_layers: List of hidden layer sizes

    Returns:
        MultiConditionalCritic model
    """
    critic = MultiConditionalCritic(
        input_dim=input_dim,
        condition_dim=condition_dim,
        hidden_layers=hidden_layers
    )

    # Build the model by calling it once
    dummy_expression = tf.zeros((1, input_dim))
    dummy_conditions = tf.zeros((1, condition_dim))
    _ = critic((dummy_expression, dummy_conditions))

    return critic


def gradient_penalty(critic, real_samples, fake_samples, conditions, lambda_gp=10.0):
    """
    Compute gradient penalty for WGAN-GP

    Args:
        critic: The critic model
        real_samples: Real expression data
        fake_samples: Generated expression data
        conditions: Condition vectors
        lambda_gp: Weight for gradient penalty

    Returns:
        Gradient penalty loss
    """
    batch_size = tf.shape(real_samples)[0]

    # Generate random interpolation weights
    alpha = tf.random.uniform((batch_size, 1), 0.0, 1.0)

    # Interpolate between real and fake samples
    # Original (mathematically equivalent but potentially numerically different):
    # interpolated = real_samples + alpha * (fake_samples - real_samples)
    # Standard WGAN-GP formulation:
    interpolated = alpha * real_samples + (1.0 - alpha) * fake_samples

    # Calculate critic scores for interpolated samples
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        critic_interpolated = critic((interpolated, conditions), training=True)

    # Calculate gradients
    gradients = tape.gradient(critic_interpolated, interpolated)

    # Calculate gradient norm
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-10)

    # Calculate gradient penalty
    gp = lambda_gp * tf.reduce_mean(tf.square(gradients_norm - 1.0))

    return gp