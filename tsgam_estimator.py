from dataclasses import dataclass, field
from itertools import combinations
from numpy import ndarray
import numpy as np
import cvxpy
from numpy.random import RandomState
from scipy import stats
from scipy.sparse import spdiags
from sklearn.base import RegressorMixin, BaseEstimator, check_array, check_is_fitted
from sklearn.utils import check_X_y, check_random_state
from spcqe import make_basis_matrix
from spcqe.functions import initialize_arrays
import pandas as pd

@dataclass
class TsgamMultiHarmonicConfig:
    num_harmonics: list[int]
    periods: list[float]
    reg_weight: float = 1.0e-4

@dataclass
class TsgamSplineConfig:
    n_knots: int | None
    lags: list[int] = field(default_factory=lambda:[0])
    reg_weight: float = 1.0e-4
    diff_reg_weight: float = 1.0
    knots: list[float] = field(default_factory=list)

@dataclass
class TsgamLinearConfig:
    lags: list[int] = field(default_factory=lambda:[0])
    reg_weight: float = 1.0e-4
    diff_reg_weight: float = 1.0

@dataclass
class TsgamArConfig:
    lags: list[int]
    l1_constraint: float


@dataclass
class TsgamSolverConfig:
    solver: str = 'CLARABEL'
    verbose: bool = True

@dataclass
class TsgamEstimatorConfig:
    multi_harmonic_config: TsgamMultiHarmonicConfig | None
    exog_config: list[TsgamSplineConfig | TsgamLinearConfig] | None
    ar_config: TsgamArConfig | None
    solver_config: TsgamSolverConfig
    random_state: RandomState | None
    freq: str  # Required: pandas frequency string (e.g., 'h' for hourly, 'H' also accepted)
    debug: bool = False


class TsgamEstimator(BaseEstimator, RegressorMixin):
    """
    Tsgam estimator.
    """
    def __init__(self, config: TsgamEstimatorConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def _extract_timestamps(self, X):
        """
        Extract timestamps from X.

        Parameters
        ----------
        X : array-like or DataFrame
            Input data. If DataFrame with DatetimeIndex, extracts index.
            If DataFrame, checks first column for datetime.
            Otherwise raises ValueError.

        Returns
        -------
        timestamps : DatetimeIndex
            Extracted timestamps.
        """
        if isinstance(X, pd.DataFrame):
            # Check if index is DatetimeIndex
            if isinstance(X.index, pd.DatetimeIndex):
                return X.index
            # Check if first column is datetime
            elif len(X.columns) > 0 and pd.api.types.is_datetime64_any_dtype(X.iloc[:, 0]):
                return pd.DatetimeIndex(X.iloc[:, 0])
            else:
                raise ValueError(
                    "X must have DatetimeIndex or first column must be datetime. "
                    "Got DataFrame without datetime index or datetime column."
                )
        else:
            raise ValueError(
                "X must be a pandas DataFrame with DatetimeIndex or datetime column. "
                f"Got {type(X)} instead."
            )

    def _timestamps_to_indices(self, timestamps, reference):
        """
        Convert timestamps to numeric indices (hours since reference).

        Parameters
        ----------
        timestamps : DatetimeIndex
            Timestamps to convert.
        reference : Timestamp
            Reference timestamp (time 0).

        Returns
        -------
        indices : ndarray
            Numeric indices in hours since reference.
        """
        return (timestamps - reference).total_seconds() / 3600.0

    def _validate_frequency(self, timestamps, expected_freq):
        """
        Validate that timestamps match expected frequency.

        Parameters
        ----------
        timestamps : DatetimeIndex
            Timestamps to validate.
        expected_freq : str
            Expected pandas frequency string (e.g., 'h' for hourly, 'H' also accepted).

        Raises
        ------
        ValueError
            If frequency doesn't match or timestamps are not regular.
        """
        if len(timestamps) < 2:
            return  # Can't validate frequency with < 2 samples

        # Normalize 'H' to 'h' for backward compatibility
        normalized_freq = expected_freq.lower() if expected_freq == 'H' else expected_freq

        # Infer frequency from timestamps
        inferred_freq = pd.infer_freq(timestamps)

        if inferred_freq is None:
            raise ValueError(
                f"Could not infer frequency from timestamps. "
                f"Timestamps must be regularly spaced with frequency '{expected_freq}'."
            )

        # Normalize inferred frequency for comparison ('H' -> 'h')
        normalized_inferred = inferred_freq.lower() if inferred_freq == 'H' else inferred_freq

        if normalized_inferred != normalized_freq:
            # Try to create expected range and compare
            try:
                expected_range = pd.date_range(
                    start=timestamps[0],
                    periods=len(timestamps),
                    freq=normalized_freq
                )
                if not timestamps.equals(expected_range):
                    raise ValueError(
                        f"Timestamps frequency '{inferred_freq}' does not match "
                        f"expected frequency '{expected_freq}'."
                    )
            except Exception as e:
                raise ValueError(
                    f"Timestamps frequency '{inferred_freq}' does not match "
                    f"expected frequency '{expected_freq}': {e}"
                )

    def _ensure_timestamp_index(self, X):
        """
        Ensure X has proper timestamp index/column, extracting timestamps.

        Parameters
        ----------
        X : array-like or DataFrame
            Input data.

        Returns
        -------
        timestamps : DatetimeIndex
            Extracted timestamps.
        X_array : ndarray
            X as array without timestamp column if it was extracted.
        """
        timestamps = self._extract_timestamps(X)

        # If X is DataFrame and first column was datetime, remove it
        if isinstance(X, pd.DataFrame) and not isinstance(X.index, pd.DatetimeIndex):
            if pd.api.types.is_datetime64_any_dtype(X.iloc[:, 0]):
                X_array = X.iloc[:, 1:].values
            else:
                X_array = X.values
        else:
            X_array = X.values if isinstance(X, pd.DataFrame) else X

        return timestamps, X_array

    def _make_regularization_matrix(self, num_harmonics,
                                   weight: float,
                                   periods: list[float],
                                   drop_constant: bool = False,
                                   standing_wave=False,
                                   trend=False,
                                   max_cross_k=None,
                                   custom_basis=None):
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

        if drop_constant:
            first_block = first_block[1:]

        coeff_i = np.concatenate(first_block + blocks_original + blocks_cross)

        D = spdiags(coeff_i, 0, coeff_i.size, coeff_i.size)
        return D



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

    def _running_view(self, arr, window, lag=1, axis=-1):
        """
        Create running view of array for AR terms.

        Parameters
        ----------
        arr : array-like
            Input array.
        window : int
            Window size (number of AR lags).
        lag : int, default=1
            Lag offset (typically 1 for standard AR).
        axis : int, default=-1
            Axis along which to create running view.

        Returns
        -------
        view : ndarray
            Running view with extra dimension of shape (len(arr), window).
        """
        mod_arr = np.r_[np.ones(window + lag - 1) * np.nan, arr[:-1]]
        shape = list(mod_arr.shape)
        shape[axis] -= (window - 1)
        assert shape[axis] > 0, f"Array too short for window={window}, lag={lag}"
        return np.lib.stride_tricks.as_strided(
            mod_arr,
            shape=shape + [window],
            strides=mod_arr.strides + (mod_arr.strides[axis],)
        )

    def _build_exog_Hs(self, exog_cfg: TsgamSplineConfig | TsgamLinearConfig, exog_var: ndarray, knots: ndarray | None = None):
        """
        Build basis matrices for an exogenous variable with lead/lag.

        This is a helper method that can be reused in both fit and predict.

        Parameters
        ----------
        exog_cfg : TsgamSplineConfig or TsgamLinearConfig
            Configuration for the exogenous variable.
        exog_var : ndarray
            Single exogenous variable column (shape: (n_samples,)).
        knots : ndarray or None, default=None
            Knot locations for spline (if None and spline config, will be computed or error).

        Returns
        -------
        Hs : list of ndarray
            List of basis matrices, one for each lag in exog_cfg.lags.
        """
        Hs = []

        for lag in exog_cfg.lags:
            if isinstance(exog_cfg, TsgamSplineConfig):
                if knots is None:
                    raise ValueError("knots must be provided for TsgamSplineConfig")
                H0 = self._make_H(exog_var, knots, include_offset=False)
                H_lag = self._make_offset_H(H0, lag)
            else:  # TsgamLinearConfig
                H0 = exog_var
                H_lag = self._make_offset_H(H0, lag)

            Hs.append(H_lag)

        return Hs

    def _process_exog_config(self, exog_cfg: TsgamSplineConfig | TsgamLinearConfig, exog_var: ndarray, knots: ndarray | None = None):
        """
        Process an exogenous variable configuration to build basis matrices.

        Parameters
        ----------
        exog_cfg : TsgamSplineConfig or TsgamLinearConfig
            Configuration for the exogenous variable.
        exog_var : ndarray
            Single exogenous variable column (shape: (n_samples,)).
        knots : ndarray or None, optional
            Pre-computed knots to use (for prediction). If None, computes from config or data.

        Returns
        -------
        valid_mask : ndarray
            Boolean mask indicating valid samples (no NaN from lead/lag operations).
        Hs : list of ndarray
            List of basis matrices, one for each lag in exog_cfg.lags.
        """
        # Get knots if spline config
        if knots is None:
            if isinstance(exog_cfg, TsgamSplineConfig):
                # Empty list means knots not specified, need to compute from n_knots
                if not exog_cfg.knots:  # Handles both None and empty list
                    if exog_cfg.n_knots:
                        knots = np.linspace(np.min(exog_var), np.max(exog_var), exog_cfg.n_knots)
                    else:
                        raise ValueError("Either knots or n_knots must be provided for TsgamSplineConfig")
                else:
                    knots = np.asarray(exog_cfg.knots)
            else:
                knots = None

        # Build Hs using helper method
        # Ensure knots is a numpy array if provided
        if knots is not None:
            knots = np.asarray(knots)
        Hs = self._build_exog_Hs(exog_cfg, exog_var, knots)

        # Find valid samples (no NaN from lead/lag operations)
        valid_mask = np.all(np.all(~np.isnan(np.asarray(Hs)), axis=-1), axis=0)

        return valid_mask, Hs

    def _get_min_samples_required(self):
        """
        Calculate minimum number of samples required based on lags.

        For positive lags (looking back), we need at least that many samples.
        For negative lags (leads/looking forward), we need at least abs(lag) samples.
        For AR lags, we need at least max(ar_lags) samples.

        Returns
        -------
        min_samples : int
            Minimum number of samples required.
        """
        all_exog_lags = []
        for exog_cfg in self.config.exog_config or []:
            all_exog_lags.extend(exog_cfg.lags)

        max_positive_lag = 0
        min_negative_lag = 0
        for lag in all_exog_lags:
            if lag > 0:
                max_positive_lag = max(max_positive_lag, lag)
            elif lag < 0:
                min_negative_lag = min(min_negative_lag, lag)
        max_negative_lag = abs(min_negative_lag)

        # AR lags are typically positive (looking back)
        max_ar_lag = 0
        if self.config.ar_config is not None:
            max_ar_lag = max(self.config.ar_config.lags) if self.config.ar_config.lags else 0

        # Need enough samples for the maximum backward-looking lag (exog or AR)
        # plus enough samples for the maximum forward-looking lag
        # plus 1 for at least one valid sample where all requirements overlap
        min_samples = max(max_positive_lag, max_ar_lag) + max_negative_lag + 1

        return min_samples

    def fit(self, X, y, sample_weight=None):
        # Extract timestamps before check_X_y converts DataFrame to array
        timestamps, X_array = self._ensure_timestamp_index(X)

        # Validate frequency
        self._validate_frequency(timestamps, self.config.freq)

        # Store frequency and reference timestamp
        # Normalize 'H' to 'h' for consistency
        normalized_freq = self.config.freq.lower() if self.config.freq == 'H' else self.config.freq
        self.freq_ = normalized_freq
        self.time_reference_ = timestamps[0]

        # Convert timestamps to numeric indices (hours since reference)
        time_indices = self._timestamps_to_indices(timestamps, self.time_reference_)
        self.time_indices_ = time_indices

        # Now validate X and y with the array version
        X_array, y = check_X_y(X_array, y,
            ensure_min_features=len(self.config.exog_config or []),
            ensure_min_samples=self._get_min_samples_required())


        self.variables_ = {
            'constant': cvxpy.Variable(),
        }
        self.exog_knots_ = []  # Store knots only when auto-computed from training data
        model_term = self.variables_['constant']
        regularization_term = 0
        self.combined_valid_mask_ = np.ones(len(y), dtype=bool)

        if self.config.exog_config:
            for ix, exog_cfg in enumerate(self.config.exog_config):
                valid_mask, Hs = self._process_exog_config(exog_cfg, X_array[:, ix])

                # Store knots only if auto-computed (not provided in config)
                if isinstance(exog_cfg, TsgamSplineConfig):
                    # Empty list means knots not specified, need to compute from n_knots
                    if not exog_cfg.knots:  # Handles both None and empty list
                        if exog_cfg.n_knots:
                            knots = np.linspace(np.min(X_array[:, ix]), np.max(X_array[:, ix]), exog_cfg.n_knots)
                            # Store auto-computed knots (need to reuse for prediction)
                            self.exog_knots_.append(knots)
                        else:
                            raise ValueError("Either knots or n_knots must be provided for TsgamSplineConfig")
                    else:
                        # Knots provided in config, don't need to store
                        self.exog_knots_.append(None)
                else:
                    self.exog_knots_.append(None)

                # Create CVXPY variable for coefficients
                # Shape: (basis_dim, num_lags)
                basis_dim = Hs[0].shape[1]
                num_lags = len(exog_cfg.lags)
                exog_coef = cvxpy.Variable((basis_dim, num_lags))

                self.variables_[f'exog_coef_{ix}'] = exog_coef
                regularization_term += cvxpy.sum_squares(exog_coef) * exog_cfg.reg_weight
                if len(exog_cfg.lags) > 1:
                    regularization_term += cvxpy.sum_squares(cvxpy.diff(exog_coef, axis=1)) * exog_cfg.diff_reg_weight
                self.combined_valid_mask_ &= valid_mask

            for ix, exog_cfg in enumerate(self.config.exog_config):
                # Rebuild Hs to build model term (Hs are only needed during fit)
                valid_mask, Hs = self._process_exog_config(exog_cfg, X_array[:, ix])
                # Sum over lags: H @ exog_coef[:, lag_ix] for each lag
                model_term += cvxpy.sum(expr=[H[self.combined_valid_mask_] @ self.variables_[f'exog_coef_{ix}'][:, lag_ix] for lag_ix, H in enumerate(Hs)])



        if self.config.multi_harmonic_config:
            # Generate basis matrix for max index + 1, then index with time_indices
            # This ensures correct phase alignment (as shown in notebook)
            max_idx = int(np.max(time_indices))
            F_full = make_basis_matrix(
                num_harmonics=self.config.multi_harmonic_config.num_harmonics,
                length=max_idx + 1,
                periods=self.config.multi_harmonic_config.periods
            )
            # Index with time_indices to get correct rows
            F = F_full[time_indices.astype(int), 1:]  # Drop constant column

            Wf = self._make_regularization_matrix(
                num_harmonics=self.config.multi_harmonic_config.num_harmonics,
                weight=1.0,
                periods=self.config.multi_harmonic_config.periods,
                drop_constant=True
            )
            self.variables_['fourier_coef'] = cvxpy.Variable(F.shape[1])
            regularization_term += self.config.multi_harmonic_config.reg_weight * cvxpy.sum_squares(Wf @ self.variables_['fourier_coef'])
            model_term += F[self.combined_valid_mask_] @ self.variables_['fourier_coef']


        error = cvxpy.sum_squares(y[self.combined_valid_mask_] - model_term) / np.sum(self.combined_valid_mask_)
        self.problem_ = cvxpy.Problem(cvxpy.Minimize(error + regularization_term))
        self.problem_.solve(solver=self.config.solver_config.solver, verbose=self.config.solver_config.verbose)

        # Fit AR model if configured
        if self.config.ar_config is not None:
            self._fit_ar_model(X_array, y, time_indices)

        return self

    def _fit_ar_model(self, X_array, y, time_indices):
        """
        Fit AR model on baseline residuals.

        Parameters
        ----------
        X_array : ndarray
            Exogenous variables array.
        y : ndarray
            Target values.
        time_indices : ndarray
            Time indices for Fourier basis.
        """
        # Get baseline predictions
        baseline_pred = np.full(len(y), self.variables_['constant'].value)

        # Add exogenous terms if present
        if self.config.exog_config:
            for ix, exog_cfg in enumerate(self.config.exog_config):
                exog_var = X_array[:, ix]
                stored_knots = self.exog_knots_[ix] if isinstance(exog_cfg, TsgamSplineConfig) else None
                _, Hs = self._process_exog_config(exog_cfg, exog_var, knots=stored_knots)
                exog_coef = self.variables_[f'exog_coef_{ix}'].value
                if exog_coef is not None:
                    exog_pred = np.sum([H @ exog_coef[:, lag_ix] for lag_ix, H in enumerate(Hs)], axis=0)
                    baseline_pred += exog_pred

        # Add Fourier terms if present
        if self.config.multi_harmonic_config:
            max_idx = int(np.max(time_indices))
            F_full = make_basis_matrix(
                num_harmonics=self.config.multi_harmonic_config.num_harmonics,
                length=max_idx + 1,
                periods=self.config.multi_harmonic_config.periods
            )
            F = F_full[time_indices.astype(int), 1:]  # Drop constant column
            fourier_coef = self.variables_['fourier_coef'].value
            if fourier_coef is not None:
                baseline_pred += F @ fourier_coef

        # Compute residuals on valid samples
        residuals = y[self.combined_valid_mask_] - baseline_pred[self.combined_valid_mask_]

        # Build AR design matrix
        if self.config.ar_config is None:
            return
        ar_config = self.config.ar_config
        ar_lags = len(ar_config.lags)
        B = self._running_view(residuals, ar_lags)
        ar_valid_mask = np.all(~np.isnan(B), axis=1)

        if self.config.debug:
            self._B_running_view_ = B
            self._ar_valid_mask_ = ar_valid_mask
            self._baseline_residuals_ = residuals

        if not np.any(ar_valid_mask):
            # Not enough data for AR model
            self.ar_coef_ = None
            self.ar_intercept_ = None
            self.ar_noise_loc_ = None
            self.ar_noise_scale_ = None
            return

        # Fit AR model using CVXPY
        theta = cvxpy.Variable(ar_lags)
        constant = cvxpy.Variable()

        ar_problem = cvxpy.Problem(
            cvxpy.Minimize(cvxpy.sum_squares(residuals[ar_valid_mask] - B[ar_valid_mask] @ theta - constant)),
            [cvxpy.norm1(theta) <= ar_config.l1_constraint]
        )
        ar_problem.solve(solver=self.config.solver_config.solver, verbose=self.config.solver_config.verbose)

        if ar_problem.status not in ["infeasible", "unbounded"]:
            assert theta.value is not None, "AR coefficients should be set"
            assert constant.value is not None, "AR intercept should be set"
            self.ar_coef_ = theta.value
            self.ar_intercept_ = constant.value

            # Fit Laplace distribution to AR model residuals
            ar_model = B[ar_valid_mask] @ theta.value + constant.value
            ar_residuals = residuals[ar_valid_mask] - ar_model
            self.ar_noise_loc_, self.ar_noise_scale_ = stats.laplace.fit(ar_residuals)
        else:
            # AR model failed to solve
            self.ar_coef_ = None
            self.ar_intercept_ = None
            self.ar_noise_loc_ = None
            self.ar_noise_scale_ = None

    def predict(self, X):
        check_is_fitted(self, ['problem_', 'time_reference_', 'freq_'])

        # Extract timestamps and validate
        timestamps, X_array = self._ensure_timestamp_index(X)

        # Validate frequency matches
        self._validate_frequency(timestamps, self.freq_)

        # Convert timestamps to indices using stored reference
        time_indices = self._timestamps_to_indices(timestamps, self.time_reference_)

        # Validate X_array shape
        X_array = check_array(X_array, ensure_min_features=len(self.config.exog_config or []))

        # Initialize prediction with constant term
        predictions = np.full(len(X_array), self.variables_['constant'].value)

        # Add exogenous terms if present
        if self.config.exog_config:
            for ix, exog_cfg in enumerate(self.config.exog_config):
                exog_var = X_array[:, ix]

                # Get stored knots if available (auto-computed during fit), otherwise None
                stored_knots = self.exog_knots_[ix] if isinstance(exog_cfg, TsgamSplineConfig) else None

                # Use _process_exog_config with stored knots (will use config knots if stored_knots is None)
                _, Hs_pred = self._process_exog_config(exog_cfg, exog_var, knots=stored_knots)

                # Compute exogenous prediction
                exog_coef = self.variables_[f'exog_coef_{ix}'].value
                if exog_coef is None:
                    raise ValueError(f"Exogenous coefficients for variable {ix} are None. Model may not have converged.")
                exog_pred = np.sum([H @ exog_coef[:, lag_ix] for lag_ix, H in enumerate(Hs_pred)], axis=0)
                predictions += exog_pred

        # Add Fourier terms if present
        if self.config.multi_harmonic_config:
            # Generate basis matrix for max index + 1, then index with time_indices
            max_idx = int(np.max(time_indices))
            min_idx = int(np.min(time_indices))

            # Handle negative indices (prediction before fit period)
            # Generate basis matrix from 0 to max_idx, then adjust indices
            if min_idx < 0:
                # Generate enough basis matrix to cover negative indices
                # We'll shift indices to be non-negative
                offset = -min_idx
                adjusted_indices = time_indices.astype(int) + offset
                basis_length = max_idx + offset + 1
            else:
                adjusted_indices = time_indices.astype(int)
                basis_length = max_idx + 1

            F_full = make_basis_matrix(
                num_harmonics=self.config.multi_harmonic_config.num_harmonics,
                length=basis_length,
                periods=self.config.multi_harmonic_config.periods
            )
            # Index with adjusted_indices to get correct rows
            F = F_full[adjusted_indices, 1:]  # Drop constant column

            fourier_coef = self.variables_['fourier_coef'].value
            if fourier_coef is None:
                raise ValueError("Fourier coefficients are None. Model may not have converged.")
            predictions += F @ fourier_coef

        return predictions

    def sample(self, X, n_samples=1, random_state=None):
        """
        Generate sample predictions with AR noise rollout.

        Parameters
        ----------
        X : array-like or DataFrame
            Input data with timestamps.
        n_samples : int, default=1
            Number of samples to generate.
        random_state : int, RandomState instance or None, default=None
            Random state for reproducible results.

        Returns
        -------
        samples : ndarray of shape (n_samples, n_pred_samples)
            Sample predictions in log scale. If AR model is fitted, includes AR noise rollout.
            Otherwise, adds small Laplace noise.
        """
        check_is_fitted(self, ['problem_', 'time_reference_', 'freq_'])
        random_state = check_random_state(random_state)

        # Get baseline predictions
        baseline_pred = self.predict(X)

        if self.config.ar_config is not None and hasattr(self, 'ar_coef_') and self.ar_coef_ is not None:
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
        """
        Generate samples with AR noise rollout using residuals.

        Parameters
        ----------
        baseline_pred : ndarray
            Baseline predictions in log scale.
        n_samples : int
            Number of samples to generate.
        random_state : RandomState
            Random state for reproducible results.

        Returns
        -------
        samples : ndarray of shape (n_samples, len(baseline_pred))
            Sample predictions with AR noise in log scale.
        """
        assert self.ar_coef_ is not None and self.ar_intercept_ is not None, \
            "AR coefficients must be set before generating samples"
        assert self.ar_noise_loc_ is not None and self.ar_noise_scale_ is not None, \
            "AR noise distribution parameters must be set before generating samples"

        ar_coef = self.ar_coef_
        ar_intercept = self.ar_intercept_
        ar_lags = len(ar_coef)

        samples = np.zeros((n_samples, len(baseline_pred)))
        for i in range(n_samples):
            # Initialize window with random noise (as in notebook)
            window = stats.laplace.rvs(
                loc=self.ar_noise_loc_,
                scale=self.ar_noise_scale_,
                size=ar_lags,
                random_state=random_state
            )
            # Generate AR noise with burn-in period (matching notebook)
            # Notebook generates length + ar_lags * 2 values, then uses last length values
            length = len(baseline_pred)
            nvals = length + ar_lags * 2
            gen_data = np.empty(nvals, dtype=float)
            for it in range(nvals):
                # Generate AR value: ar_coef @ window + intercept + noise
                ar_val = ar_coef @ window + ar_intercept + stats.laplace.rvs(
                    loc=self.ar_noise_loc_,
                    scale=self.ar_noise_scale_,
                    random_state=random_state
                )
                gen_data[it] = ar_val
                # Update window: roll and replace last element
                window = np.roll(window, -1)
                window[-1] = ar_val
            # Use last length values (after burn-in)
            ar_noise = gen_data[-length:]
            samples[i] = baseline_pred + ar_noise
        return samples



if __name__ == "__main__":
    """
    Baseline configuration replicating the notebook baseline model.

    Configuration:
    - Multi-harmonic: [6, 4, 3] harmonics for periods [365.2425*24, 7*24, 24]
    - Temperature spline: 10 knots, lags [-3, -2, -1, 0, 1, 2, 3]
    - Regularization: 1e-4 for Fourier and exog, 1.0 for exog diff
    - Solver: CLARABEL with verbose=True
    - No AR model (baseline only)
    """
    import pandas as pd
    from pathlib import Path

    # Load data from same place as notebook
    def load_notebook_data(sheet='RI', years=[2020, 2021]):
        df_list = []
        for year in years:
            fp = Path('.') / 'ISO_Data' / f'{year}_smd_hourly.xlsx'
            df = pd.read_excel(fp, sheet_name=sheet)
            df['year'] = year
            df.index = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Hr_End'].map(lambda x: f"{x-1}:00:00")) + pd.Timedelta(hours=1)
            df_list.append(df)
        return pd.concat(df_list, axis=0)

    # Load data
    print("Loading data...")
    df = load_notebook_data(sheet='RI', years=[2020, 2021])

    # Prepare y (log-transformed RT_Demand) and X (temperature only)
    df_subset = df.loc["2020":"2021"]
    y = np.log(df_subset["RT_Demand"]).values
    X = pd.DataFrame({'temp': df_subset["Dry_Bulb"].values}, index=df_subset.index)

    # Multi-harmonic configuration for time features
    multi_harmonic_config = TsgamMultiHarmonicConfig(
        num_harmonics=[6, 4, 3],
        periods=[365.2425 * 24, 7 * 24, 24]
    )

    # Spline configuration for temperature (exogenous variable)
    exog_config: list[TsgamSplineConfig | TsgamLinearConfig] = [
        TsgamSplineConfig(
            knots=[],  # Empty list means knots will be auto-generated from data
            n_knots=10,  # Number of knots to generate
            lags=[-3, -2, -1, 0, 1, 2, 3],
            reg_weight=1e-4,  # Regularization weight for coefficients
            diff_reg_weight=1.0  # Regularization weight for differences between lags
        )
    ]

    # No AR model in baseline (AR is added later in the notebook)
    ar_config = None

    # Solver configuration
    solver_config = TsgamSolverConfig(
        solver='CLARABEL',
        verbose=True
    )

    # Create main config
    config = TsgamEstimatorConfig(
        multi_harmonic_config=multi_harmonic_config,
        exog_config=exog_config,
        ar_config=ar_config,
        solver_config=solver_config,
        random_state=None,
        freq='h',  # Hourly frequency (lowercase 'h' preferred, 'H' also accepted)
        debug=False
    )

    # Create estimator
    print("\nCreating estimator...")
    estimator = TsgamEstimator(config=config)

    print("\nConfiguration:")
    if config.multi_harmonic_config:
        print(f"  Multi-harmonic: {config.multi_harmonic_config.num_harmonics} harmonics")
        print(f"  Periods: {config.multi_harmonic_config.periods}")
    if config.exog_config:
        print(f"  Exog config: {len(config.exog_config)} exogenous variable(s)")
        for ix, exog_cfg in enumerate(config.exog_config):
            if isinstance(exog_cfg, TsgamSplineConfig):
                print(f"    [{ix}] Type: Spline")
                print(f"        n_knots: {exog_cfg.n_knots}")
            else:
                print(f"    [{ix}] Type: Linear")
            print(f"        lags: {exog_cfg.lags}")
            print(f"        reg_weight: {exog_cfg.reg_weight}")
            print(f"        diff_reg_weight: {exog_cfg.diff_reg_weight}")
    print(f"  Solver: {config.solver_config.solver} (verbose={config.solver_config.verbose})")

    # Fit the model
    print("\nFitting model...")
    estimator.fit(X, y)

    print("\nFitting complete!")
    print(f"Problem status: {estimator.problem_.status}")
    if estimator.problem_.status in ["optimal", "optimal_inaccurate"]:
        print(f"Optimal value: {estimator.problem_.value:.6e}")
