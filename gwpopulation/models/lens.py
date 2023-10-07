"""
Implemented lens models
"""
import inspect

from ..cupy_utils import trapz, xp
from ..utils import powerlaw, truncnorm

def power_law_mu(dataset, gamma, mumin, mumax):
    r"""
    Power law model for magnification distribution.

    .. math::

        p_{\text{pow}}(\mu) &\propto \mu^{-\alpha} : \mu_\min \leq \mu < \mu_\max

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mu' (:math:`\mu`).
    gamma: float
        Negative power law exponent for the magnification.
    mumin: float
        Minimum magnification (:math:`\mu_\min`).
    mmax: float
        Maximum magnification (:math:`\mu_\max`).
    """
    return powerlaw(dataset["mu"], -gamma, mumax, mumin)
