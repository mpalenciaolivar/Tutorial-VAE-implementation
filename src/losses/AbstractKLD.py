# coding: utf-8
from abc import abstractmethod
from typing import Any


class AbstractKLD:
    @staticmethod
    @abstractmethod
    def compute(q, p):
        """Computes the Kullback-Leibler divergence between two distributions.

        Args:
            q (dict): Variational distribution parameters
            p (dict): Prior distribution parameters
        """
        pass
