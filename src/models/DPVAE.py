# coding: utf-8
import tensorflow_probability as tfp
from .GaussianVAE import GaussianVAE


class DPVAE(GaussianVAE):
    """A VAE-inferred Dirichlet Process. Instead if Nalisnick & Smyth's formulation, we use the implicit reparameterization gradients:
    
    Nalisnick, Eric, and Padhraic Smyth. "Stick-breaking variational autoencoders." arXiv preprint arXiv:1605.06197 (2016).
    Figurnov, Mikhail, Shakir Mohamed, and Andriy Mnih. "Implicit reparameterization gradients." Advances in neural information processing systems 31 (2018).
    """
    def _reparameterize(self):
        concentration1 = self.distribution_parameters["param1"]
        concentration0 = self.distribution_parameters["param2"]
        
