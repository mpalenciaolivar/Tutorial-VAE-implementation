# coding: utf-8
import tensorflow as tf
from MLPFactory import MLPFactory


class GaussianVAE:
    """A VAE-inferred Gaussian distribution, according to Kingma & Welling's paper:   
    Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).
    """
    def __init__(self, latent_dim, encoder=None, decoder=None):
        """Inits the VAE according to the specified latent dimension, encoder and decoder.

        Args:
            latent_dim (int): The dimension for the latent variable.
            encoder (tfk.Sequential, optional): Model for the encoding part. Do **not** specify the stochastic layer. Defaults to None.
            decoder (tfk.Sequential, optional): Model for the decoding part. Defaults to None.
        """
        if encoder is None:
            self.encoder = MLPFactory.generate()

        if decoder is None:
            self.decoder = MLPFactory.generate()
        self.latent_dim = latent_dim
        self.encoder = self._add_stochastic_layer(self.encoder)
    
    def _add_stochastic_layer(self, encoder):
        # VAEs have stochastic layer by design. We need to add it.
        # We multiply the latent dim by 2 because the Gaussian distribution
        # includes 2 parameters. We'll split the Dense layer elsewhere.
        tfk = tf.keras
        encoder.add(tfk.Dense(2 * self.latent_dim))
        return encoder

    def _encode(self, X):
        param1, param2 = tf.split(self.encoder(X),
                                  num_or_size_splits=2,
                                  axis=1)
        self.distribution_parameters = {"param1": param1, "param2": param2}
    
    def _reparameterize(self):
        mu = self.distribution_parameters["param1"]
        logvar = self.distribution_parameters["param2"]
        eps = tf.random.normal(shape=mu.shape)
        sample = eps * tf.exp(logvar * .5) + mu
        return sample
    
    def _decode(self, sample):
        logits = self.decoder(sample)
        return logits
    
    @tf.function
    def sample_from_distribution(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        probs = tf.sigmoid(self._decode(eps))
        return probs
    
    def __call__(self, X, get_probs=False):
        self._encode(X)
        sample = self._reparameterize()
        logits = self._decode(sample)
        if get_probs:
            probs = tf.sigmoid(logits)
            return probs
        return logits
