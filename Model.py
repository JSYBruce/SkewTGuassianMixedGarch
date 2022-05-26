# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:50:06 2022

@author: BruceKing
"""

from typing import Optional, Union, cast
import numpy as np

from customtyping import Float64Array
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from typing import Callable, List, Optional, Sequence, Tuple, Union
import warnings

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
from scipy.special import comb, gamma, gammainc, gammaincc
from DoubleNormal import DoubleNormal
from Garch import GARCH
from util import ensure1d


from customtyping import (
    ArrayLike,
    ArrayLike1D,
    DateLike,
    Float64Array,
    ForecastingMethod,
    Label,
    Literal,
)
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
_callback_info = {"iter": 0, "llf": 0.0, "count": 0, "display": 1}

class ARCHModel():
    """
    Abstract base class for mean models in ARCH processes.  Specifies the
    conditional mean process.

    All public methods that raise NotImplementedError should be overridden by
    any subclass.  Private methods that raise NotImplementedError are optional
    to override but recommended where applicable.
    """

    def __init__(
        self,
        y: Optional[ArrayLike] = None,
        volatility: Optional[GARCH] = None,
        distribution: Optional[DoubleNormal] = None,
        hold_back: Optional[int] = None,
        rescale: Optional[bool] = None,
    ) -> None:
        self._name = "ARCHModel"
        self._is_pandas = isinstance(y, (pd.DataFrame, pd.Series))
        if y is not None:
            self._y_series = cast(pd.Series, ensure1d(y, "y", series=True))
        else:
            self._y_series = cast(pd.Series, ensure1d(np.empty((0,)), "y", series=True))
        self._y = np.asarray(self._y_series)
        if not np.all(np.isfinite(self._y)):
            raise ValueError(
                "NaN or inf values found in y. y must contains only finite values."
            )
        self._y_original = y
        self.volatility = volatility
        self.distribution = distribution
        self._fit_indices: List[int] = [0, int(self._y.shape[0])]
        self._fit_y = self._y

        self.hold_back: Optional[int] = hold_back
        self._hold_back = 0 if hold_back is None else hold_back

        self.rescale: Optional[bool] = rescale
        self.scale: float = 1.0

        self._backcast: Union[None, float, Float64Array] = None
        self._var_bounds: Optional[Float64Array] = None
        print("Hello, ArchModel")
    @property
    def name(self) -> str:
        """The name of the model."""
        return self._name

    def constraints(self) -> Tuple[Float64Array, Float64Array]:
        """
        Construct linear constraint arrays  for use in non-linear optimization

        Returns
        -------
        a : ndarray
            Number of constraints by number of parameters loading array
        b : ndarray
            Number of constraints array of lower bounds

        Notes
        -----
        Parameters satisfy a.dot(parameters) - b >= 0
        """
        return np.empty((0, self.num_params)), np.empty(0)

    def bounds(self) -> List[Tuple[float, float]]:
        """
        Construct bounds for parameters to use in non-linear optimization

        Returns
        -------
        bounds : list (2-tuple of float)
            Bounds for parameters to use in estimation.
        """
        num_params = self.num_params
        return [(-np.inf, np.inf)] * num_params

    @property
    def y(self) -> Optional[ArrayLike]:
        """Returns the dependent variable"""
        return self._y_original


    def _check_scale(self, resids: Float64Array) -> None:
        check = self.rescale in (None, True)
        if not check:
            return
        orig_scale = scale = resids.var()
        rescale = 1.0
        while not 0.1 <= scale < 10000.0 and scale > 0:
            if scale < 1.0:
                rescale *= 10
            else:
                rescale /= 10
            scale = orig_scale * rescale**2
        if rescale == 1.0:
            return
        self.scale = rescale


    def resids(
        self,
        params: Float64Array,
        y: Optional[Float64Array] = None,
        regressors: Optional[Float64Array] = None,
    ) -> Float64Array:
        """
        Compute model residuals

        Parameters
        ----------
        params : ndarray
            Model parameters
        y : ndarray, optional
            Alternative values to use when computing model residuals
        regressors : ndarray, optional
            Alternative regressor values to use when computing model residuals

        Returns
        -------
        resids : ndarray
            Model residuals
        """
        return self.y - params

    def _loglikelihood(
        self,
        parameters: Float64Array,
        sigma1: Float64Array,
        sigma2: Float64Array,
        backcast: Union[float, Float64Array],
        var_bounds: Float64Array,
        individual: bool = False,
    ) -> Union[float, Float64Array]:
        """
        Computes the log-likelihood using the entire model

        Parameters
        ----------
        parameters
        sigma2
        backcast
        individual : bool, optional

        Returns
        -------
        neg_llf : float
            Negative of model loglikelihood
        """
        # Parse parameters
        _callback_info["count"] += 1

        # 1. Resids
        mp, vp, dp, weight, skewtparams = self._parse_parameters(parameters)
        resids1 = self.resids(mp[0])
        resids2 = self.resids(mp[1])
        ###########
        # 2. Compute sigma2 using VolatilityModel
        sigma1, sigma2 = self.volatility.compute_variance(
            vp, resids1, resids2, sigma1, sigma2, backcast, var_bounds
        )
        # 3. Compute log likelihood using Distribution
        llf = self.distribution.loglikelihood(np.append(mp,skewtparams), resids1, resids2, sigma1, sigma2, weight, individual)
        if not individual:
            _callback_info["llf"] = llf_f = -float(llf)
            return llf_f

        return cast(np.ndarray, -llf)

    def _parse_parameters(
        self, x: ArrayLike
    ) -> Tuple[Float64Array, Float64Array, Float64Array]:
        """Return the parameters of each model in a tuple"""
        x = np.asarray(x, dtype=np.float64)
        km, kv = int(self.num_params), int(self.volatility.num_params)
      
        return x[:km], x[km : km + kv], [], x[km + kv: km + kv + 1], x[km + kv+1: km + kv + 3]

    def fit(
        self,
        update_freq: int = 1,
        starting_values: ArrayLike1D = None,
        show_warning: bool = True,
        first_obs: Union[int, DateLike] = None,
        last_obs: Union[int, DateLike] = None,
        tol: Optional[float] = None,
        options: Optional[Dict[str, Any]] = None,
        backcast: Union[None, float, Float64Array] = None,
    ):
        r"""
        Estimate model parameters

        Parameters
        ----------
        update_freq : int, optional
            Frequency of iteration updates.  Output is generated every
            `update_freq` iterations. Set to 0 to disable iterative output.
        disp : {bool, "off", "final"}
            Either 'final' to print optimization result or 'off' to display
            nothing. If using a boolean, False is "off" and True is "final"
        starting_values : ndarray, optional
            Array of starting values to use.  If not provided, starting values
            are constructed by the model components.
        show_warning : bool, optional
            Flag indicating whether convergence warnings should be shown.
        first_obs : {int, str, datetime, Timestamp}
            First observation to use when estimating model
        last_obs : {int, str, datetime, Timestamp}
            Last observation to use when estimating model
        tol : float, optional
            Tolerance for termination.
        options : dict, optional
            Options to pass to `scipy.optimize.minimize`.  Valid entries
            include 'ftol', 'eps', 'disp', and 'maxiter'.
        backcast : {float, ndarray}, optional
            Value to use as backcast. Should be measure :math:`\sigma^2_0`
            since model-specific non-linear transformations are applied to
            value before computing the variance recursions.pa

        Returns
        -------
        results : ARCHModelResult
            Object containing model results

        Notes
        -----
        A ConvergenceWarning is raised if SciPy's optimizer indicates
        difficulty finding the optimum.

        Parameters are optimized using SLSQP.
        """
        if self._y_original is None:
            raise RuntimeError("Cannot estimate model without data.")
        # 1. Check in ARCH or Non-normal dist.  If no ARCH and normal,
        # use closed form
        v, d = self.volatility, self.distribution
        self.num_params = 2
        d.num_params = 0
        offsets = np.array((self.num_params, v._num_params, d.num_params), dtype=int)
        print("offsets",offsets)
        total_params = sum(offsets)
        print(total_params)
        resids1 = self.resids(self.starting_values())
        resids2 = self.resids(self.starting_values())
        if self.scale != 1.0:
            # Scale changed, rescale data and reset model
            self._y = cast(np.ndarray, self.scale * np.asarray(self._y_original))
            self._adjust_sample(first_obs, last_obs)
            resids = self.resids(self.starting_values())

        if backcast is None:
            backcast = v.backcast(resids1)
        else:
            assert backcast is not None
            backcast = v.backcast_transform(backcast)

        assert backcast is not None

        sigma1 = np.zeros_like(resids1)
        sigma2 = np.zeros_like(resids2)
        self._backcast = backcast
        sv_volatility = v.starting_values(resids1, resids2)
        self._var_bounds = var_bounds = v.variance_bounds(resids1)
        sigma1, sigma2 = v.compute_variance(sv_volatility, resids1, resids2, sigma1, sigma2, backcast, var_bounds)
        std_resids = resids1 / np.sqrt(sigma2)
        
        # 2. Construct constraint matrices from all models and distribution
        bounds = self.bounds()
        bounds.extend(v.bounds(resids1))
        bounds.extend([(0,1)])
        bounds.extend([(2.05, 500.0)])
        bounds.extend([(-1, 1)])
        # 3. Construct starting values from all models
        sv = starting_values
            
        # 4. Estimate models using constrained optimizatio
        func = self._loglikelihood
        args = (sigma1, sigma2, backcast, var_bounds)

        from scipy.optimize import NonlinearConstraint
        con = lambda x: (x[3] + x[5])*x[7] + (x[4] + x[6])*(1 - x[7])
        ineq_constraints = NonlinearConstraint(con, 0, 1)
        
        from scipy.optimize import minimize

        options = {} if options is None else options
        
        
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "Values in x were outside bounds during a minimize step",
                RuntimeWarning,
            )
            opt = minimize(
                func,
                sv,
                args=args,
                method="SLSQP",
                bounds=bounds,
                constraints=ineq_constraints,
                tol=tol,
                callback=_callback,
                options=options,
            )

        
        # 5. Return results
        params = opt.x
        loglikelihood = -1.0 * opt.fun

        mp, vp, dp, weight, nu = self._parse_parameters(params)


        self.mu = mp
        
        resids1 = self.resids(mp[0])
        resids2 = self.resids(mp[1])
        
        sigma1, sigma2 = self.volatility.compute_variance(
            vp, resids1, resids2, sigma1, sigma2, backcast, var_bounds
        )
        
        np.savetxt("sigma1.csv", sigma1, delimiter=",")
        np.savetxt("sigma2.csv", sigma2, delimiter=",")
        import pickle
        with open("sigma1_data", "wb") as fp:   #Pickling
            pickle.dump(sigma1, fp)
        
        with open("sigma2_data", "wb") as fp:   #Pickling
            pickle.dump(sigma2, fp)
            
        with open("resids1", "wb") as fp:   #Pickling
            pickle.dump(resids1, fp)
        
        with open("resids2", "wb") as fp:   #Pickling
            pickle.dump(resids2, fp)
         
        vol = weight * sigma1 + (1 - weight) * sigma2
        # Reshape resids and vol
        first_obs, last_obs = self._fit_indices
        resids_final = np.empty_like(self._y, dtype=np.float64)
        resids_final.fill(np.nan)
        resids_final[first_obs:last_obs] = weight * resids1 + (1 - weight) * resids2
        vol_final = sqrt(vol)

        names = ["mean1","mean2", "omega", "alpha1", "alpha2","beta1", "beta2", "weight", "eta", "lambda"]
        fit_start, fit_stop = self._fit_indices
        from copy import deepcopy
        r2 = self._r2(mp[0])
        model_copy = deepcopy(self)
        print('params',len(params))
        return (
            params,
            r2,
            resids_final,
            vol_final,
            "Robust",
            self._y_series,
            names,
            loglikelihood,
            self._is_pandas,
            opt,
            fit_start,
            fit_stop,
            model_copy,
        )
    
    
    def _r2(self, params: Float64Array) -> float:
        y = self._fit_y
        constant = False
        if constant:
            y = y - np.mean(y)
        tss = float(y.dot(y))
        
        e = self.resids(params)
            
        return 1.0 - float(e.T.dot(e)) / tss
    
    def starting_values(self) -> Float64Array:
        """
        Returns starting values for the mean model, often the same as the
        values returned from fit

        Returns
        -------
        sv : ndarray
            Starting values
        """
        return 0.0
    
    def compute_param_cov(
        self,
        params: Float64Array,
        backcast: Union[None, float, Float64Array] = None,
        robust: bool = True,
    ) -> Float64Array:
        """
        Computes parameter covariances using numerical derivatives.

        Parameters
        ----------
        params : ndarray
            Model parameters
        backcast : float
            Value to use for pre-sample observations
        robust : bool, optional
            Flag indicating whether to use robust standard errors (True) or
            classic MLE (False)

        """
        resids = self.resids(self.starting_values())
        var_bounds = self.volatility.variance_bounds(resids)
        nobs = resids.shape[0]
        if backcast is None and self._backcast is None:
            backcast = self.volatility.backcast(resids)
            self._backcast = backcast
        elif backcast is None:
            backcast = self._backcast

        kwargs = {
            "sigma1": np.zeros_like(resids),
            "sigma2": np.zeros_like(resids),
            "backcast": backcast,
            "var_bounds": var_bounds,
            "individual": False,
        }

        hess = approx_hess(params, self._loglikelihood, kwargs=kwargs)
        hess /= nobs
        inv_hess = np.linalg.inv(hess)
        if robust:
            kwargs["individual"] = True
            scores = approx_fprime(
                params, self._loglikelihood, kwargs=kwargs
            )  # type: np.ndarray
            score_cov = np.cov(scores.T)
            return inv_hess.dot(score_cov).dot(inv_hess) / nobs
        else:
            return inv_hess / nobs
    
    
    
    def starting_values(self) -> Float64Array:
        """
        Returns starting values for the mean model, often the same as the
        values returned from fit

        Returns
        -------
        sv : ndarray
            Starting values
        """
        return 0.0
    
    def compute_param_cov(
        self,
        params: Float64Array,
        backcast: Union[None, float, Float64Array] = None,
        robust: bool = True,
    ) -> Float64Array:
        """
        Computes parameter covariances using numerical derivatives.

        Parameters
        ----------
        params : ndarray
            Model parameters
        backcast : float
            Value to use for pre-sample observations
        robust : bool, optional
            Flag indicating whether to use robust standard errors (True) or
            classic MLE (False)

        """
        resids = self.resids(self.starting_values())
        var_bounds = self.volatility.variance_bounds(resids)
        nobs = resids.shape[0]
        if backcast is None and self._backcast is None:
            backcast = self.volatility.backcast(resids)
            self._backcast = backcast
        elif backcast is None:
            backcast = self._backcast

        kwargs = {
            "sigma1": np.zeros_like(resids),
            "sigma2": np.zeros_like(resids),
            "backcast": backcast,
            "var_bounds": var_bounds,
            "individual": False,
        }

        hess = approx_hess(params, self._loglikelihood, kwargs=kwargs)
        hess /= nobs
        inv_hess = np.linalg.inv(hess)
        if robust:
            kwargs["individual"] = True
            scores = approx_fprime(
                params, self._loglikelihood, kwargs=kwargs
            )  # type: np.ndarray
            score_cov = np.cov(scores.T)
            return inv_hess.dot(score_cov).dot(inv_hess) / nobs
        else:
            return inv_hess / nobs
    

def constraint(a: Float64Array, b: Float64Array) -> List[Dict[str, object]]:
    """
    Generate constraints from arrays

    Parameters
    ----------
    a : ndarray
        Parameter loadings
    b : ndarray
        Constraint bounds

    Returns
    -------
    constraints : dict
        Dictionary of inequality constraints, one for each row of a

    Notes
    -----
    Parameter constraints satisfy a.dot(parameters) - b >= 0
    """

    def factory(coeff: Float64Array, val: float) -> Callable[..., float]:
        def f(params: Float64Array, *args: Any) -> float:
            return np.dot(coeff, params) - val

        return f

    constraints = []
    for i in range(a.shape[0]):
        con = {"type": "ineq", "fun": factory(a[i], b[i])}
        constraints.append(con)

    return constraints

def implicit_constant(x: Float64Array) -> bool:
    """
    Test a matrix for an implicit constant

    Parameters
    ----------
    x : ndarray
        Array to be tested

    Returns
    -------
    constant : bool
        Flag indicating whether the array has an implicit constant - whether
        the array has a set of columns that adds to a constant value
    """
    nobs = x.shape[0]
    rank = np.linalg.matrix_rank(np.hstack((np.ones((nobs, 1)), x)))
    return rank == x.shape[1]


def _callback(parameters: Float64Array, *args: Any) -> None:
    """
    Callback for use in optimization

    Parameters
    ----------
    parameters : ndarray
        Parameter value (not used by function).
    *args
        Any other arguments passed to the minimizer.

    Notes
    -----
    Uses global values to track iteration, iteration display frequency,
    log likelihood and function count
    """

    _callback_info["iter"] += 1
    disp = "Iteration: {0:>6},   Func. Count: {1:>6.3g},   Neg. LLF: {2}"
    if _callback_info["iter"] % _callback_info["display"] == 0:
        print(
            disp.format(
                _callback_info["iter"], _callback_info["count"], _callback_info["llf"]
            )
        )
