# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:53:08 2022

@author: BruceKing
"""
from property_cached import cached_property
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
import scipy.stats as stats
import pandas as pd
import datetime as dt

from statsmodels.iolib.summary import Summary, fmt_2cols, fmt_params
import numpy as np
from scipy.optimize import OptimizeResult
from statsmodels.iolib.table import SimpleTable

from typing import Callable, List, Optional, Sequence, Tuple, Union

from Model import ARCHModel

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

from customtyping import ArrayLike, ArrayLike1D, Float64Array


class _SummaryRepr(object):
    """Base class for returning summary as repr and str"""

    def summary(self) -> Summary:
        raise NotImplementedError("Subclasses must implement")

    def __repr__(self) -> str:
        out = self.__str__() + "\n"
        out += self.__class__.__name__
        out += ", id: {0}".format(hex(id(self)))
        return out

    def __str__(self) -> str:
        return self.summary().as_text()


class ARCHModelFixedResult(_SummaryRepr):
    """
    Results for fixed parameters for an ARCHModel model

    Parameters
    ----------
    params : ndarray
        Estimated parameters
    resid : ndarray
        Residuals from model.  Residuals have same shape as original data and
        contain nan-values in locations not used in estimation
    volatility : ndarray
        Conditional volatility from model
    dep_var : Series
        Dependent variable
    names : list (str)
        Model parameter names
    loglikelihood : float
        Loglikelihood at specified parameters
    is_pandas : bool
        Whether the original input was pandas
    model : ARCHModel
        The model object used to estimate the parameters
    """

    def __init__(
        self,
        params: Float64Array,
        resid: Float64Array,
        volatility: Float64Array,
        dep_var: pd.Series,
        names: Sequence[str],
        loglikelihood: float,
        is_pandas: bool,
        model: ARCHModel,
    ) -> None:
        self._params = params
        self._resid = resid
        self._is_pandas = is_pandas
        self.model = model
        self._datetime = dt.datetime.now()
        self._dep_var = dep_var
        self._dep_name = dep_var.name
        self._names = names
        self._loglikelihood = loglikelihood
 
        self._nobs = self.model._fit_y.shape[0]
        self._index = dep_var.index
        self._volatility = volatility

    def summary(self) -> Summary:
        """
        Constructs a summary of the results from a fit model.

        Returns
        -------
        summary : Summary instance
            Object that contains tables and facilitated export to text, html or
            latex
        """
        # Summary layout
        # 1. Overall information
        # 2. Mean parameters
        # 3. Volatility parameters
        # 4. Distribution parameters
        # 5. Notes

        model = self.model
        model_name = model.name + " - " + model.volatility.name

        # Summary Header
        top_left = [
            ("Dep. Variable:", self._dep_name),
            ("Mean Model:", model.name),
            ("Vol Model:", model.volatility.name),
            ("Distribution:", model.distribution.name),
            ("Method:", "User-specified Parameters"),
            ("", ""),
            ("Date:", self._datetime.strftime("%a, %b %d %Y")),
            ("Time:", self._datetime.strftime("%H:%M:%S")),
        ]

        top_right = [
            ("R-squared:", "--"),
            ("Adj. R-squared:", "--"),
            ("Log-Likelihood:", "%#10.6g" % self.loglikelihood),
            ("AIC:", "%#10.6g" % self.aic),
            ("BIC:", "%#10.6g" % self.bic),
            ("No. Observations:", self._nobs),
            ("", ""),
            ("", ""),
        ]

        title = model_name + " Model Results"
        stubs = []
        vals = []
        for stub, val in top_left:
            stubs.append(stub)
            vals.append([val])
        table = SimpleTable(vals, txt_fmt=fmt_2cols, title=title, stubs=stubs)

        # create summary table instance
        smry = Summary()
        # Top Table
        # Parameter table
        fmt = fmt_2cols
        fmt["data_fmts"][1] = "%18s"

        top_right = [("%-21s" % ("  " + k), v) for k, v in top_right]
        stubs = []
        vals = []
        for stub, val in top_right:
            stubs.append(stub)
            vals.append([val])
        table.extend_right(SimpleTable(vals, stubs=stubs))
        smry.tables.append(table)

        stubs = list(self._names)
        header = ["coef"]
        coef_vals = (self.params,)
        formats = [(10, 4)]
        pos = 0
        param_table_data = []
        for _ in range(len(coef_vals[0])):
            row = []
            for i, val in enumerate(coef_vals):
                if isinstance(val[pos], np.float64):
                    converted = format_float_fixed(val[pos], *formats[i])
                else:
                    converted = val[pos]
                row.append(converted)
            pos += 1
            param_table_data.append(row)

        mc = self.model.num_params
        vc = self.model.volatility.num_params
        dc = self.model.distribution.num_params
        counts = (mc, vc, dc)
        titles = ("Mean Model", "Volatility Model", "Distribution")
        total = 0
        for title, count in zip(titles, counts):
            if count == 0:
                continue

            table_data = param_table_data[total : total + count]
            table_stubs = stubs[total : total + count]
            total += count
            table = SimpleTable(
                table_data,
                stubs=table_stubs,
                txt_fmt=fmt_params,
                headers=header,
                title=title,
            )
            smry.tables.append(table)

        extra_text = [
            "Results generated with user-specified parameters.",
            "Std. errors not available when the model is not estimated, ",
        ]
        smry.add_extra_txt(extra_text)
        return smry
    
    
def format_float_fixed(x: float, max_digits: int = 10, decimal: int = 4) -> str:
    """Formats a floating point number so that if it can be well expressed
    in using a string with digits len, then it is converted simply, otherwise
    it is expressed in scientific notation"""
    # basic_format = '{:0.' + str(digits) + 'g}'
    if x == 0:
        return ("{:0." + str(decimal) + "f}").format(0.0)
    scale = np.log10(np.abs(x))
    scale = np.sign(scale) * np.ceil(np.abs(scale))
    if scale > (max_digits - 2 - decimal) or scale < -(decimal - 2):
        formatted = ("{0:" + str(max_digits) + "." + str(decimal) + "e}").format(x)
    else:
        formatted = ("{0:" + str(max_digits) + "." + str(decimal) + "f}").format(x)
    return formatted


import copy
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from statsmodels.iolib.summary import Summary, fmt_2cols, fmt_params
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult
from statsmodels.tsa.tsatools import lagmat
class ARCHModelResult(ARCHModelFixedResult):
    """
    Results from estimation of an ARCHModel model

    Parameters
    ----------
    params : ndarray
        Estimated parameters
    param_cov : {ndarray, None}
        Estimated variance-covariance matrix of params.  If none, calls method
        to compute variance from model when parameter covariance is first used
        from result
    r2 : float
        Model R-squared
    resid : ndarray
        Residuals from model.  Residuals have same shape as original data and
        contain nan-values in locations not used in estimation
    volatility : ndarray
        Conditional volatility from model
    cov_type : str
        String describing the covariance estimator used
    dep_var : Series
        Dependent variable
    names : list (str)
        Model parameter names
    loglikelihood : float
        Loglikelihood at estimated parameters
    is_pandas : bool
        Whether the original input was pandas
    optim_output : OptimizeResult
        Result of log-likelihood optimization
    fit_start : int
        Integer index of the first observation used to fit the model
    fit_stop : int
        Integer index of the last observation used to fit the model using
        slice notation `fit_start:fit_stop`
    model : ARCHModel
        The model object used to estimate the parameters
    """

    def __init__(
        self,
        params: Float64Array,
        param_cov: Optional[Float64Array],
        r2: float,
        resid: Float64Array,
        volatility: Float64Array,
        cov_type: str,
        dep_var: pd.Series,
        names: Sequence[str],
        loglikelihood: float,
        is_pandas: bool,
        optim_output: OptimizeResult,
        fit_start: int,
        fit_stop: int,
        model: ARCHModel,
    ) -> None:
        super().__init__(
            params, resid, volatility, dep_var, names, loglikelihood, is_pandas, model
        )
        
        self._fit_indices = (fit_start, fit_stop)
        self._param_cov = param_cov
        self._r2 = r2
        self.cov_type: str = cov_type
        self._optim_output = optim_output
        self._nobs = self.model._fit_y.shape[0]
        import pickle
        with open("sigma1_data", "rb") as fp:   #Pickling
            self.sigma1 = pickle.load(fp)
        
        with open("sigma2_data", "rb") as fp:   #Pickling
            self.sigma2 = pickle.load(fp)
            
        with open("resids1", "rb") as fp:   #Pickling
            self.resids1 = pickle.load(fp)
        
        with open("resids2", "rb") as fp:   #Pickling
            self.resids2 = pickle.load(fp)

    @cached_property
    def scale(self) -> float:
        """
        The scale applied to the original data before estimating the model.

        If scale=1.0, the the data have not been rescaled.  Otherwise, the
        model parameters have been estimated on scale * y.
        """
        return self.model.scale

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Parameter confidence intervals

        Parameters
        ----------
        alpha : float, optional
            Size (prob.) to use when constructing the confidence interval.

        Returns
        -------
        ci : DataFrame
            Array where the ith row contains the confidence interval  for the
            ith parameter
        """
        cv = stats.norm.ppf(1.0 - alpha / 2.0)
        se = self.std_err
        params = self.params

        return pd.DataFrame(
            np.vstack((params - cv * se, params + cv * se)).T,
            columns=["lower", "upper"],
            index=self._names,
        )

    def summary(self) -> Summary:
        """
        Constructs a summary of the results from a fit model.

        Returns
        -------
        summary : Summary instance
            Object that contains tables and facilitated export to text, html or
            latex
        """
        # Summary layout
        # 1. Overall information
        # 2. Mean parameters
        # 3. Volatility parameters
        # 4. Distribution parameters
        # 5. Notes

        model = self.model
        model_name = model.name + " - " + model.volatility.name

        # Summary Header
        top_left = [
            ("Dep. Variable:", self._dep_name),
            ("Mean Model:", model.name),
            ("Vol Model:", model.volatility.name),
            ("Distribution:", model.distribution.name),
            ("Method:", "Maximum Likelihood"),
            ("", ""),
            ("Date:", self._datetime.strftime("%a, %b %d %Y")),
            ("Time:", self._datetime.strftime("%H:%M:%S")),
        ]

        top_right = [
            ("R-squared:", "%#8.3f" % self.rsquared),
            ("Adj. R-squared:", "%#8.3f" % self.rsquared_adj),
            ("Log-Likelihood:", "%#10.6g" % self.loglikelihood),
            ("AIC:", "%#10.6g" % self.aic),
            ("BIC:", "%#10.6g" % self.bic),
            ("No. Observations:", self._nobs),
            ("Df Residuals:", self.nobs - self.model.num_params),
            ("Df Model:", self.model.num_params),
        ]

        title = model_name + " Model Results"
        stubs = []
        vals = []
        for stub, val in top_left:
            stubs.append(stub)
            vals.append([val])
        table = SimpleTable(vals, txt_fmt=fmt_2cols, title=title, stubs=stubs)

        # create summary table instance
        smry = Summary()
        # Top Table
        # Parameter table
        fmt = fmt_2cols
        fmt["data_fmts"][1] = "%18s"

        top_right = [("%-21s" % ("  " + k), v) for k, v in top_right]
        stubs = []
        vals = []
        for stub, val in top_right:
            stubs.append(stub)
            vals.append([val])
        table.extend_right(SimpleTable(vals, stubs=stubs))
        smry.tables.append(table)

        conf_int = np.asarray(self.conf_int())
        conf_int_str = []
        for c in conf_int:
            conf_int_str.append(
                "["
                + format_float_fixed(c[0], 7, 3)
                + ","
                + format_float_fixed(c[1], 7, 3)
                + "]"
            )

        stubs = list(self._names)
        header = ["coef", "std err", "t", "P>|t|", "95.0% Conf. Int."]
        table_vals = (
            self.params,
            self.std_err,
            self.tvalues,
            self.pvalues,
            conf_int_str,
        )
        # (0,0) is a dummy format
        formats = [(10, 4), (9, 3), (9, 3), (9, 3), (0, 0)]
        pos = 0
        param_table_data = []
        for _ in range(len(table_vals[0])):
            row = []
            for i, val in enumerate(table_vals):
                if isinstance(val[pos], np.float64):
                    converted = format_float_fixed(val[pos], *formats[i])
                else:
                    converted = val[pos]
                row.append(converted)
            pos += 1
            param_table_data.append(row)

        mc = self.model.num_params
        vc = self.model.volatility.num_params + 2
        dc = self.model.distribution.num_params
        counts = (mc, vc, dc)
        titles = ("Mean Model", "Volatility Model", "Distribution")
        total = 0
        for title, count in zip(titles, counts):
            if count == 0:
                continue

            table_data = param_table_data[total : total + count]
            table_stubs = stubs[total : total + count]
            total += count
            table = SimpleTable(
                table_data,
                stubs=table_stubs,
                txt_fmt=fmt_params,
                headers=header,
                title=title,
            )
            smry.tables.append(table)

        extra_text = ["Covariance estimator: " + self.cov_type]

        return smry
    
    @cached_property
    def rsquared(self) -> float:
        """
        R-squared
        """
        return self._r2
    
    @cached_property
    def fit_start(self) -> int:
        """Start of sample used to estimate parameters"""
        return self._fit_indices[0]

    @cached_property
    def fit_stop(self) -> int:
        """End of sample used to estimate parameters"""
        return self._fit_indices[1]

    @cached_property
    def rsquared_adj(self) -> float:
        """
        Degree of freedom adjusted R-squared
        """
        return 1 - (
            (1 - self.rsquared) * (self.nobs - 1) / (self.nobs - self.model.num_params)
        )

    @cached_property
    def pvalues(self) -> pd.Series:
        """
        Array of p-values for the t-statistics
        """
        return pd.Series(
            stats.norm.sf(np.abs(self.tvalues)) * 2, index=self._names, name="pvalues"
        )

    @cached_property
    def std_err(self) -> pd.Series:
        """
        Array of parameter standard errors
        """
        return pd.Series(
            np.sqrt(np.diag(self.param_cov)), index=self._names, name="std_err"
        )

    @cached_property
    def tvalues(self) -> pd.Series:
        """
        Array of t-statistics testing the null that the coefficient are 0
        """
        tvalues = self.params / self.std_err
        tvalues.name = "tvalues"
        return tvalues

    @cached_property
    def convergence_flag(self) -> int:
        """
        scipy.optimize.minimize result flag
        """
        return self._optim_output.status

    
    
    @cached_property
    def model(self) -> ARCHModel:
        """
        Model instance used to produce the fit
        """
        return self._model

    @cached_property
    def loglikelihood(self) -> float:
        """Model loglikelihood"""
        return self._loglikelihood

    @cached_property
    def aic(self) -> float:
        """Akaike Information Criteria

        -2 * loglikelihood + 2 * num_params"""
        return -2 * self.loglikelihood + 2 * self.num_params

    @cached_property
    def num_params(self) -> int:
        """Number of parameters in model"""
        return len(self.params)

    @cached_property
    def bic(self) -> float:
        """
        Schwarz/Bayesian Information Criteria

        -2 * loglikelihood + log(nobs) * num_params
        """
        return -2 * self.loglikelihood + np.log(self.nobs) * self.num_params

    @cached_property
    def params(self) -> pd.Series:
        """Model Parameters"""
        return pd.Series(self._params, index=self._names, name="params")

    @cached_property
    def nobs(self) -> int:
        """
        Number of data points used to estimate model
        """
        return self._nobs
    
    
    @cached_property
    def param_cov(self) -> pd.DataFrame:
        """Parameter covariance"""
        print("param_cov")
        self.cov_type = "robust"
        if self._param_cov is not None:
            print("param_cov_None")
            param_cov = self._param_cov
        else:
            params = np.asarray(self.params)
            if self.cov_type == "robust":
                print("robust")
                param_cov = self.model.compute_param_cov(params)
            else:
                print("not robust")
                param_cov = self.model.compute_param_cov(params, robust=False)
        return pd.DataFrame(param_cov, columns=self._names, index=self._names)
    