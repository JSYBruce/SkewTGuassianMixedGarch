# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from customtyping import Float64Array, Int32Array, ArrayLike, ArrayLike1D, Float64Array
from typing import Callable, List, Optional, Sequence, Tuple, Union
from scipy.special import gamma
from numpy import (
    abs,
    array,
    asarray,
    empty,
    exp,
    int64,
    integer,
    isscalar,
    log,
    nan,
    ndarray,
    ones_like,
    pi,
    sign,
    sqrt,
    sum,
)
from scipy.special import comb, gamma, gammainc, gammaincc, gammaln
import numpy as np

class DoubleNormal():
    """
    t + normal distribution for use with ARCH models

    Parameters
    ----------
    """

    def __init__(
        self) -> None:
        self._name = "SkewT+Gaussian"
        self.name = "SkewT+Gaussian"
        self.num_params = 3
    def constraints(self) -> Tuple[Float64Array, Float64Array]:
        return array([[1, 0], [-1, 0], [0, 1], [0, -1]]), array([2.05, -300.0, -1, -1])


    def bounds(self, resids: Float64Array) -> List[Tuple[float, float]]:
        return []

    def loglikelihood(
        self,
        parameters: Union[Sequence[float], ArrayLike1D],
        resids1: ArrayLike,
        resids2: ArrayLike,
        sigma1: ArrayLike,
        sigma2: ArrayLike,
        weight: Float64Array,
        individual: bool = False,
    ) -> Union[float, Float64Array]:
        r"""Computes the log-likelihood of assuming residuals are normally
        distributed, conditional on the variance

        Parameters
        ----------
        parameters : ndarray
            The normal likelihood has no shape parameters. Empty since the
            standard normal has no shape parameters.
        resids  : ndarray
            The residuals to use in the log-likelihood calculation
        sigma2 : ndarray
            Conditional variances of resids
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ll : float
            The log-likelihood

        Notes
        -----
        The log-likelihood of a single data point x is

        .. math::

            \ln f\left(x\right)=-\frac{1}{2}\left(\ln2\pi+\ln\sigma^{2}
            +\frac{x^{2}}{\sigma^{2}}\right)

        """
        
        eta = parameters[2]
        lam = parameters[3]
        const_c = exp(float(
            gammaln((eta + 1) / 2) - gammaln(eta / 2) - log(pi * (eta - 2)) / 2
        ))
        const_a = float(4 * lam * const_c) * (eta - 2) / (eta - 1)
        const_b = (1 + 3 * lam ** 2 - const_a ** 2) ** 0.5
        resids = (resids1) / sigma1 ** 0.5
        skewt_pdf = const_b * const_c / (sigma1**0.5)
        if abs(lam) >= 1.0:
            lam = sign(lam) * (1.0 - 1e-6)
        llf_resid = (
            (const_b * resids + const_a) / (1 + sign(resids + const_a / const_b) * lam)
        ) ** 2
        skewt_pdf *= (1 + llf_resid / (eta - 2))**(-(eta + 1) / 2 )
        
        
        lls = log( weight * skewt_pdf + (1-weight)/sqrt(2*pi*sigma2)*exp(-0.5*(resids2)**2 /sigma2) )
        if individual:
            return lls
        else:
            return sum(lls)

    def starting_values(self, std_resid: Float64Array) -> Float64Array:
        return empty(0)
    

    