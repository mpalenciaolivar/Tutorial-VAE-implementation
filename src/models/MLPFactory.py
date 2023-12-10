# coding: utf-8
import tensorflow as tf


tfk = tf.keras
tfkl = tfk.layers


class MLPFactory:
    """Generates MLP according to a configuration.

    Returns:
        A TF-Keras Sequential model.
    """
    @staticmethod
    def generate(hidden_layers=[100, 100]):
        model = tfk.Sequential()
        for layer in hidden_layers:
            model.add(tfkl.Dense(layer))
        return model
