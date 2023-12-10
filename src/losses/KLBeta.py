# coding: utf-8
import tensorflow_probability as tfp
from .AbstractKLD import AbstractKLD


class KLBeta(AbstractKLD):
    def compute(q, p):
        """Computes the Kullback-Leibler divergence between two Beta distributions.

        Args:
            q (dict): Variational distribution parameters
            p (dict): Prior distribution parameters
        """
        pass