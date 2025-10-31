"""
Load forecasting estimator compatible with scikit-learn.

This module provides a LoadForecastRegressor that combines multi-Fourier time features,
cubic spline or linear exogenous variables with lead/lag, and optional AR residual modeling.
"""

from typing import Any, Callable, Optional
import numpy as np
import cvxpy as cvx
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.utils import check_random_state
from scipy.sparse import spdiags
from itertools import combinations
import scipy.stats as stats
from spcqe import make_basis_matrix
from spcqe.functions import initialize_arrays

class LoadForecastRegressor(BaseEstimator, RegressorMixin):
    """
    Load forecasting regressor with multi-Fourier, exogenous, and AR components.

    This estimator fits a load forecasting model that combines:
    - Multi-Fourier basis functions for seasonal patterns
    - Cubic spline or linear basis functions for exogenous variables with lead/lag
    - Optional autoregressive modeling of residuals

    Parameters
    ----------
    num_harmonics : list of int, default=[1]
        Number of harmonics for each Fourier period.
    periods : list of float, default=[24]
        Periods for Fourier basis functions (in hours).
    exog_mode : {'spline', 'linear'}, default='linear'
        Treatment of exogenous variables. 'spline' uses cubic splines,
        'linear' uses simple linear terms.
    n_knots : int or None, default=None
        Number of knots for spline basis (only used if exog_mode='spline').
        If None, knots must be provided explicitly.
    knots : array-like or None, default=None
        Explicit knot locations for spline basis (only used if exog_mode='spline').
        If None, knots are auto-generated using n_knots.
    exog_lags : list of int, default=[0]
        Lead/lag offsets for exogenous variables.
    fourier_reg_weight : float, default=1e-4
        Regularization weight for Fourier coefficients.
    exog_reg_weight : float, default=1e-4
        Regularization weight for exogenous variable coefficients.
    exog_diff_reg_weight : float, default=1.0
        Regularization weight for differences between lag coefficients.
    fit_ar : bool, default=False
        Whether to fit AR model on residuals.
    ar_lags : int, default=1 (only used if fit_ar=True)
        Number of AR lags to use.
    ar_l1_constraint : float, default=0.95 (only used if fit_ar=True)
        L1 norm constraint for AR coefficients.
    cvxpy_solver : str, default='CLARABEL'
        CVXPY solver to use.
    cvxpy_verbose : bool, default=False
        Whether to use verbose CVXPY output.
    debug : bool, default=False
        Whether to save debug variables (e.g., _last_problem_value_,
        _baseline_residuals_, _B_running_view_, etc.) for testing/debugging.
    random_state : int, RandomState instance or None, default=None
        Random state for reproducible results.

    Attributes
    ----------
    time_coef_ : array-like
        Fitted Fourier coefficients.
    exog_coef_ : array-like
        Fitted exogenous variable coefficients.
    knots_ : array-like or None
        Knot locations used for spline basis (if exog_mode='spline').
    ar_coef_ : array-like or None
        Fitted AR coefficients (if fit_ar=True).
    ar_intercept_ : float or None
        Fitted AR intercept (if fit_ar=True).
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from load_modeling.load_model_estimator import LoadForecastRegressor
    >>>
    >>> # Generate sample data
    >>> n_samples = 1000
    >>> time_idx = np.arange(n_samples)
    >>> temperature = 20 + 10 * np.sin(2 * np.pi * time_idx / 24) + np.random.normal(0, 2, n_samples)
    >>> load = 100 + 2 * temperature + 5 * np.sin(2 * np.pi * time_idx / 24) + np.random.normal(0, 5, n_samples)
    >>>
    >>> # Prepare data (time index + exogenous variables)
    >>> X = np.column_stack([time_idx, temperature])
    >>> y = np.log(load)  # Model works in log space
    >>>
    >>> # Fit estimator
    >>> estimator = LoadForecastRegressor(exog_mode='linear', fit_ar=False)
    >>> estimator.fit(X, y)
    >>>
    >>> # Make predictions
    >>> predictions = estimator.predict(X)
    """

    def __init__(self,
                 num_harmonics=[1],
                 periods=[24],
                 exog_mode='linear',
                 n_knots=None,
                 knots=None,
                 exog_lags=[0],
                 fourier_reg_weight=1e-4,
                 exog_reg_weight=1e-4,
                 exog_diff_reg_weight=1.0,
                 fit_ar=False,
                 ar_lags=1,
                 ar_l1_constraint=0.95,
                 cvxpy_solver='CLARABEL',
                 cvxpy_verbose=False,
                 debug=False,
                 random_state=None):

        # Validate parameters
        if exog_mode not in ['spline', 'linear']:
            raise ValueError(f"exog_mode must be 'spline' or 'linear', got {exog_mode}")

        if exog_mode == 'spline' and n_knots is None and knots is None:
            raise ValueError("Either n_knots or knots must be provided when exog_mode='spline'")

        if not isinstance(exog_lags, (list, tuple, np.ndarray)):
            raise ValueError("exog_lags must be a list, tuple, or array")

        if fourier_reg_weight < 0:
            raise ValueError("fourier_reg_weight must be non-negative")

        if exog_reg_weight < 0:
            raise ValueError("exog_reg_weight must be non-negative")

        if exog_diff_reg_weight < 0:
            raise ValueError("exog_diff_reg_weight must be non-negative")

        if ar_lags <= 0:
            raise ValueError("ar_lags must be positive")

        if not 0 <= ar_l1_constraint <= 1:
            raise ValueError("ar_l1_constraint must be between 0 and 1")

        # Store parameters
        self.num_harmonics = num_harmonics
        self.periods = periods
        self.exog_mode = exog_mode
        self.n_knots = n_knots
        self.knots = knots
        self.exog_lags = exog_lags
        self.fourier_reg_weight = fourier_reg_weight
        self.exog_reg_weight = exog_reg_weight
        self.exog_diff_reg_weight = exog_diff_reg_weight
        self.fit_ar = fit_ar
        self.ar_lags = ar_lags
        self.ar_l1_constraint = ar_l1_constraint
        self.cvxpy_solver = cvxpy_solver
        self.cvxpy_verbose = cvxpy_verbose
        self.debug = debug
        self.random_state = random_state

    def _make_H(self, x, knots, include_offset=False):
        """
        Create cubic spline basis matrix.

        Parameters
        ----------
        x : array-like
            Input values.
        knots : array-like
            Knot locations.
        include_offset : bool, default=False
            Whether to include constant term.

        Returns
        -------
        H : ndarray
            Basis matrix.
        """
        def d_func(x, k, k_max):
            n1 = np.clip(np.power(x - k, 3), 0, np.inf)
            n2 = np.clip(np.power(x - k_max, 3), 0, np.inf)
            d1 = k_max - k
            out = (n1 - n2) / d1
            return out

        nK = len(knots)
        H = np.ones((len(x), nK), dtype=float)
        H[:, 1] = x
        for _i in range(nK - 2):
            _j = _i + 2
            H[:, _j] = d_func(x, knots[_i], knots[-1]) - d_func(
                x, knots[-2], knots[-1]
            )
        if include_offset:
            return H
        else:
            return H[:, 1:]

    def _make_offset_H(self, H, offset):
        """
        Create lead/lag version of basis matrix.

        Parameters
        ----------
        H : ndarray
            Original basis matrix.
        offset : int
            Lead/lag offset (positive = lag, negative = lead).

        Returns
        -------
        newH : ndarray
            Offset basis matrix with NaN padding.
        """
        newH = np.roll(np.copy(H), -offset, axis=0)
        if offset > 0:
            newH[-offset:] = np.nan
        elif offset < 0:
            newH[:-offset] = np.nan
        return newH

    def _compute_exog_prediction(self, Hs, exog_coef, valid_mask=None):
        """
        Compute exogenous variable prediction using einsum.

        Parameters
        ----------
        Hs : list of ndarray
            List of basis matrices, one per lag.
        exog_coef : ndarray
            Exogenous coefficients of shape (n_features, n_lags).
        valid_mask : ndarray or None, default=None
            Boolean mask for valid samples. If None, uses all samples.

        Returns
        -------
        exog_pred : ndarray
            Exogenous predictions.
        """
        Hs_stack = np.stack(Hs, axis=-1)  # Shape: (n_samples, n_features, n_lags)
        if valid_mask is not None:
            exog_pred = np.einsum('ifl,fl->i', Hs_stack[valid_mask], exog_coef)
            # Pad with NaNs for invalid samples
            full_pred = np.full(len(Hs_stack), np.nan)
            full_pred[valid_mask] = exog_pred
            return full_pred
        else:
            exog_pred = np.einsum('ifl,fl->i', Hs_stack, exog_coef)
            return np.nan_to_num(exog_pred, nan=0.0)

    def _running_view(self, arr, window, axis=-1):
        """
        Create running view of array for AR terms.

        Parameters
        ----------
        arr : array-like
            Input array.
        window : int
            Window size.
        axis : int, default=-1
            Axis along which to create running view.

        Returns
        -------
        view : ndarray
            Running view with extra dimension.
        """
        mod_arr = np.r_[np.ones(window) * np.nan, arr[:-1]]
        shape = list(mod_arr.shape)
        shape[axis] -= (window-1)
        assert(shape[axis] > 0)
        return np.lib.stride_tricks.as_strided(
            mod_arr,
            shape=shape + [window],
            strides=mod_arr.strides + (mod_arr.strides[axis],))

    def _solve_optimization_problem(self, problem, exog_coef):
        """
        Solve optimization problem with specified solver.

        Parameters
        ----------
        problem : cvx.Problem
            The optimization problem to solve.
        exog_coef : cvx.Variable or None
            The exogenous coefficients variable (if applicable).

        Returns
        -------
        time_coef : array
            Fitted time coefficients.
        exog_coef_value : array or None
            Fitted exogenous coefficients (or None if not applicable).
        """
        problem.solve(solver=self.cvxpy_solver, verbose=self.cvxpy_verbose)
        if problem.status in ["optimal", "optimal_inaccurate"]:
            if self.debug:
                self._last_problem_value_ = problem.value
                self._last_problem_status_ = problem.status
            time_coef = problem.variables()[0].value
            exog_coef_value = getattr(exog_coef, 'value', None)
            return time_coef, exog_coef_value
        else:
            raise RuntimeError(f"{self.cvxpy_solver} failed with status: {problem.status}")

    def _get_exog_basis_func(self) -> Callable:
        """Get the basis function for exogenous variables based on exog_mode."""
        if self.exog_mode == 'spline':
            return lambda x: self._make_H(x, self.knots_, include_offset=False)
        elif self.exog_mode == 'linear':
            return lambda x: np.column_stack([np.ones(len(x)), x])
        else:
            raise ValueError(f"exog_mode must be 'spline' or 'linear', got {self.exog_mode}")

    def _build_Hs(self, exog_vars, h_func: Optional[Callable] = None):
        """Build basis matrices for exogenous variables with lead/lag."""
        if h_func is None:
            h_func = self._get_exog_basis_func()
        Hs: list[Any] = []
        for lag in self.exog_lags:
            H_lag = []
            for exog_col in exog_vars.T:
                H0 = h_func(exog_col)
                H_lag.append(self._make_offset_H(H0, lag))
            Hs.append(np.hstack(H_lag))
        return Hs

    def _make_regularization_matrix(self, num_harmonics, weight, periods,
                                   standing_wave=False, trend=False,
                                   max_cross_k=None, custom_basis=None):
        """
        Create regularization matrix for Fourier coefficients.

        Parameters
        ----------
        num_harmonics : int or array-like
            Number of harmonics for each period.
        weight : float
            Regularization weight.
        periods : float or array-like
            Periods for each harmonic block.
        standing_wave : bool or array-like, default=False
            Whether to use standing wave basis.
        trend : bool, default=False
            Whether to include trend term.
        max_cross_k : int or None, default=None
            Maximum cross terms.
        custom_basis : dict or None, default=None
            Custom basis matrices.

        Returns
        -------
        D : sparse matrix
            Regularization matrix.
        """
        sort_idx, Ps, num_harmonics, standing_wave = initialize_arrays(
            num_harmonics, periods, standing_wave, custom_basis
        )
        ls_original = [weight * (2 * np.pi) / np.sqrt(P) for P in Ps]

        # Create sequence of values from 1 to K
        i_value_list = []
        for ix, nh in enumerate(num_harmonics):
            if standing_wave[ix]:
                i_value_list.append(np.arange(1, nh + 1))
            else:
                i_value_list.append(np.repeat(np.arange(1, nh + 1), 2))

        # Create blocks of coefficients
        blocks_original = [iv * lx for iv, lx in zip(i_value_list, ls_original)]
        if custom_basis is not None:
            for ix, val in custom_basis.items():
                ixt = np.where(sort_idx == ix)[0][0]
                blocks_original[ixt] = ls_original[ixt] * np.arange(1, val.shape[1] + 1)

        if max_cross_k is not None:
            max_cross_k *= 2

        # Compute cross-term penalties
        blocks_cross = [
            [l2 for l1 in c[0][:max_cross_k] for l2 in c[1][:max_cross_k]]
            for c in combinations(blocks_original, 2)
        ]

        # Combine blocks
        if trend is False:
            first_block = [np.zeros(1)]
        else:
            first_block = [np.zeros(2)]
        coeff_i = np.concatenate(first_block + blocks_original + blocks_cross)

        # Create diagonal matrix
        D = spdiags(coeff_i, 0, coeff_i.size, coeff_i.size)
        return D

    def fit(self, X, y, sample_weight=None):
        """
        Fit the load forecasting model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. First column should be time indices, remaining columns
            are exogenous variables.
        y : array-like of shape (n_samples,)
            Target values (should be log-transformed).
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Sample weights (not currently used).

        Returns
        -------
        self : LoadForecastRegressor
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False)

        # Store feature information
        self.n_features_in_ = X.shape[1]

        # Extract time indices and exogenous variables
        exog_vars = X[:, 1:] if X.shape[1] > 1 else None

        # Build Fourier basis matrix
        F = make_basis_matrix(
            num_harmonics=self.num_harmonics,
            length=len(y),
            periods=self.periods
        )

        # Build regularization matrix for Fourier coefficients
        # Note: notebook uses weight=1, then multiplies by 1e-4 in objective
        Wf = self._make_regularization_matrix(
            num_harmonics=self.num_harmonics,
            weight=1.0,  # Match notebook
            periods=self.periods
        )

        # Handle exogenous variables
        if exog_vars is not None:
            # Compute knots for spline mode if needed
            if self.exog_mode == 'spline':
                if self.knots is None:
                    if self.n_knots is None:
                        raise ValueError("Either knots or n_knots must be provided when exog_mode='spline'")
                    exog_min = np.min(exog_vars)
                    exog_max = np.max(exog_vars)
                    self.knots_ = np.linspace(exog_min, exog_max, self.n_knots)
                else:
                    self.knots_ = np.array(self.knots)

            Hs = self._build_Hs(exog_vars)
            # Find valid samples (no NaN from lead/lag operations)
            valid_mask = np.all(np.all(~np.isnan(np.asarray(Hs)), axis=-1), axis=0)

            # Solve optimization problem
            a = cvx.Variable(F.shape[1])  # Fourier coefficients
            c = cvx.Variable((Hs[0].shape[1], len(Hs)))  # Exogenous coefficients

            # Build exogenous term
            exog_term = cvx.sum(expr=[H[valid_mask] @ c[:, _ix] for _ix, H in enumerate(Hs)])

            # Objective function
            error = cvx.sum_squares(y[valid_mask] - F[valid_mask] @ a - exog_term) / np.sum(valid_mask)
            regularization = (self.fourier_reg_weight * cvx.sum_squares(Wf @ a) +
                            self.exog_reg_weight * cvx.sum_squares(c))

            # Add difference regularization only if we have enough lags
            if len(self.exog_lags) > 1:
                regularization += self.exog_diff_reg_weight * cvx.sum_squares(cvx.diff(c, axis=1))

            problem = cvx.Problem(cvx.Minimize(error + regularization))

            # Solve using helper method
            self.time_coef_, self.exog_coef_ = self._solve_optimization_problem(problem, c)

        else:
            # No exogenous variables
            valid_mask = np.ones(len(y), dtype=bool)
            Hs = None
            a = cvx.Variable(F.shape[1])
            error = cvx.sum_squares(y[valid_mask] - F[valid_mask] @ a) / np.sum(valid_mask)
            regularization = self.fourier_reg_weight * cvx.sum_squares(Wf @ a)

            problem = cvx.Problem(cvx.Minimize(error + regularization))

            # Solve using helper method
            self.time_coef_, self.exog_coef_ = self._solve_optimization_problem(problem, None)

        # Fit AR model on residuals if requested
        self.ar_coef_ = None
        self.ar_intercept_ = None

        if self.fit_ar:
            self._fit_ar_model(F, Hs, exog_vars, y, valid_mask)

        return self

    def _fit_ar_model(self, F, Hs, exog_vars, y, valid_mask):
        """Fit AR model on baseline residuals."""
        baseline_pred = F[valid_mask] @ self.time_coef_
        if self.exog_coef_ is not None and Hs is not None:
            exog_pred = self._compute_exog_prediction(Hs, self.exog_coef_, valid_mask)
            baseline_pred += exog_pred[valid_mask]

        residuals = y[valid_mask] - baseline_pred

        # Build AR design matrix
        B = self._running_view(residuals, self.ar_lags)
        ar_valid_mask = np.all(~np.isnan(B), axis=1)

        if self.debug:
            self._B_running_view_ = B
            self._ar_valid_mask_ = ar_valid_mask
            self._baseline_residuals_ = residuals

        if not np.any(ar_valid_mask):
            return

        theta = cvx.Variable(self.ar_lags)
        constant = cvx.Variable()

        ar_problem = cvx.Problem(
            cvx.Minimize(cvx.sum_squares(residuals[ar_valid_mask] - B[ar_valid_mask] @ theta - constant)),
            [cvx.norm1(theta) <= self.ar_l1_constraint]
        )
        ar_problem.solve(solver=self.cvxpy_solver, verbose=self.cvxpy_verbose)

        if ar_problem.status not in ["infeasible", "unbounded"]:
            assert theta.value is not None, "AR coefficients should be set"
            assert constant.value is not None, "AR intercept should be set"
            self.ar_coef_ = theta.value
            self.ar_intercept_ = constant.value

    def predict(self, X):
        """
        Make predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. First column should be time indices, remaining columns
            are exogenous variables.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """

        check_is_fitted(self, ['time_coef_', 'exog_coef_'])
        X = check_array(X, ensure_min_features=self.n_features_in_)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        # Extract time indices and exogenous variables
        exog_vars = X[:, 1:] if X.shape[1] > 1 else None

        # Build Fourier basis matrix
        F = make_basis_matrix(
            num_harmonics=self.num_harmonics,
            length=len(X),
            periods=self.periods
        )

        # Compute baseline prediction
        baseline_pred = F @ self.time_coef_

        # Add exogenous component if present
        if exog_vars is not None and self.exog_coef_ is not None:
            Hs = self._build_Hs(exog_vars)
            exog_pred = self._compute_exog_prediction(Hs, self.exog_coef_)
            baseline_pred += exog_pred

        return baseline_pred

    def sample(self, X, n_samples=1, random_state=None):
        """
        Generate sample predictions with AR noise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        n_samples : int, default=1
            Number of samples to generate.
        random_state : int, RandomState instance or None, default=None
            Random state for reproducible results.

        Returns
        -------
        samples : ndarray of shape (n_samples, n_pred_samples)
            Sample predictions.
        """

        check_is_fitted(self, ['time_coef_', 'exog_coef_', 'ar_coef_', 'ar_intercept_'])
        X = check_array(X)

        random_state = check_random_state(random_state)

        # Get baseline predictions
        baseline_pred = self.predict(X)

        if self.ar_coef_ is not None and self.ar_intercept_ is not None:
            samples = self._generate_ar_samples(baseline_pred, n_samples, random_state)
        else:
            # No AR model, just add small noise
            noise = stats.laplace.rvs(
                loc=0, scale=0.1, size=(n_samples, len(baseline_pred)),
                random_state=random_state
            )
            samples = baseline_pred + noise

        return samples

    def _generate_ar_samples(self, baseline_pred, n_samples, random_state):
        """Generate samples with AR noise."""
        assert self.ar_coef_ is not None and self.ar_intercept_ is not None, \
            "AR coefficients must be set before generating samples"
        ar_coef = self.ar_coef_
        ar_intercept = self.ar_intercept_

        samples = np.zeros((n_samples, len(baseline_pred)))
        for i in range(n_samples):
            window = stats.laplace.rvs(
                loc=0, scale=0.1, size=self.ar_lags, random_state=random_state
            )
            ar_noise = np.zeros(len(baseline_pred))
            for t in range(len(baseline_pred)):
                ar_val = ar_coef @ window + ar_intercept
                ar_noise[t] = ar_val
                window = np.roll(window, -1)
                window[-1] = ar_val
            samples[i] = baseline_pred + ar_noise
        return samples
