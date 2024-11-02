# tflib/ops/linear.py
import tensorflow as tf

def Linear(name, input_dim, output_dim, inputs, initialization=None):
    """
    Linear layer using variable scope for reuse
    """
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        if initialization is None:
            initialization = tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_avg",
                distribution="uniform"
            )
            
        weights = tf.compat.v1.get_variable(
            'Weights',
            [input_dim, output_dim],
            initializer=initialization
        )
        
        biases = tf.compat.v1.get_variable(
            'Biases',
            [output_dim],
            initializer=tf.zeros_initializer()
        )
        
        return tf.matmul(inputs, weights) + biases

# tflib/ops/conv1d.py
def Conv1D(name, input_dim, output_dim, filter_size, inputs, stride=1, padding='SAME'):
    """
    1-dimensional convolution using variable scope for reuse
    """
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        filter_shape = [filter_size, input_dim, output_dim]
        
        weights = tf.compat.v1.get_variable(
            'Weights',
            filter_shape,
            initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_avg",
                distribution="uniform"
            )
        )
        
        biases = tf.compat.v1.get_variable(
            'Biases',
            [output_dim],
            initializer=tf.zeros_initializer()
        )

        conv = tf.nn.conv1d(
            inputs,
            filters=weights,
            stride=stride,
            padding=padding,
            data_format='NCW'  # channels first format
        )
        
        return tf.nn.bias_add(conv, biases, data_format='NCW')