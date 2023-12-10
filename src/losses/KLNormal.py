# coding: utf-8
import tensorflow_probability as tfp
from .AbstractKLD import AbstractKLD


class KLNormal(AbstractKLD):
    def compute(q, p):
        """Computes the Kullback-Leibler divergence between two Gaussian distributions.

        Args:
            q (dict): Variational distribution parameters
            p (dict): Prior distribution parameters
        """
        pass
