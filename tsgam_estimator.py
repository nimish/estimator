from dataclasses import dataclass, field
from itertools import combinations
from numpy import ndarray
import numpy as np
import cvxpy
from numpy.random import RandomState
from scipy.sparse import spdiags
from sklearn.base import RegressorMixin, BaseEstimator, check_array, check_is_fitted
from sklearn.utils import check_X_y
from spcqe import make_basis_matrix
from spcqe.functions import initialize_arrays

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
    debug: bool = False


class TsgamEstimator(BaseEstimator, RegressorMixin):
    """
    Tsgam estimator.
    """
    def __init__(self, config: TsgamEstimatorConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config


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

    def _process_exog_config(self, exog_cfg: TsgamSplineConfig | TsgamLinearConfig, exog_var: ndarray):
        """
        Process an exogenous variable configuration to build basis matrices and coefficients.

        Parameters
        ----------
        exog_cfg : TsgamSplineConfig or TsgamLinearConfig
            Configuration for the exogenous variable.
        exog_var : ndarray
            Single exogenous variable column (shape: (n_samples,)).

        Returns
        -------
        valid_mask : ndarray
            Boolean mask indicating valid samples (no NaN from lead/lag operations).
        Hs : list of ndarray
            List of basis matrices, one for each lag in exog_cfg.lags.
        exog_coef : cvxpy.Variable
            CVXPY variable for exogenous coefficients.
        """
        Hs = []

        # Build basis matrices for each lag
        for lag in exog_cfg.lags:
            if isinstance(exog_cfg, TsgamSplineConfig):
                # Spline mode: use knots from config
                # For now, assume knots are provided per exogenous variable
                # If only one set of knots provided, use it; otherwise use the first
                knots = exog_cfg.knots[0] if exog_cfg.knots else None
                if knots is None:
                    # Generate knots if not provided
                    if exog_cfg.n_knots:
                        n_knots = exog_cfg.n_knots[0] if isinstance(exog_cfg.n_knots, list) else exog_cfg.n_knots
                        knots = np.linspace(np.min(exog_var), np.max(exog_var), n_knots)
                    else:
                        raise ValueError("Either knots or n_knots must be provided for TsgamSplineConfig")
                H0 = self._make_H(exog_var, knots, include_offset=False)
                H_lag = self._make_offset_H(H0, lag)
            else:  # TsgamLinearConfig
                # Linear mode: just X with lag
                H0 = exog_var
                H_lag = self._make_offset_H(H0, lag)

            Hs.append(H_lag)

        # Find valid samples (no NaN from lead/lag operations)
        valid_mask = np.all(np.all(~np.isnan(np.asarray(Hs)), axis=-1), axis=0)

        # Create CVXPY variable for coefficients
        # Shape: (basis_dim, num_lags)
        basis_dim = Hs[0].shape[1]
        num_lags = len(exog_cfg.lags)
        exog_coef = cvxpy.Variable((basis_dim, num_lags))

        return valid_mask, Hs, exog_coef

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
        # must have as many columns in X as there are exog_config
        # must have as many rows in X as there are rows in y

        X, y = check_X_y(X, y,
            ensure_min_features=len(self.config.exog_config or []),
            ensure_min_samples=self._get_min_samples_required())


        self.variables_ = {
            'constant': cvxpy.Variable(),
        }
        self.exog_terms_ = []
        model_term = self.variables_['constant']
        regularization_term = 0
        self.combined_valid_mask_ = np.ones(len(y), dtype=bool)

        if self.config.exog_config:
            for ix, exog_cfg in enumerate(self.config.exog_config):
                valid_mask, Hs, exog_coef = self._process_exog_config(exog_cfg, X[:, ix])
                self.variables_[f'exog_coef_{ix}'] = exog_coef
                regularization_term += cvxpy.sum_squares(exog_coef) * exog_cfg.reg_weight
                if len(exog_cfg.lags) > 1:
                    regularization_term += cvxpy.sum_squares(cvxpy.diff(exog_coef, axis=1)) * exog_cfg.diff_reg_weight
                self.exog_terms_.append(Hs)
                self.combined_valid_mask_ &= valid_mask

            for ix, Hs in enumerate(self.exog_terms_):
                # Sum over lags: H @ exog_coef[:, lag_ix] for each lag
                model_term += cvxpy.sum(expr=[H[self.combined_valid_mask_] @ self.variables_[f'exog_coef_{ix}'][:, lag_ix] for lag_ix, H in enumerate(Hs)])



        if self.config.multi_harmonic_config:

            F = make_basis_matrix(
                num_harmonics=self.config.multi_harmonic_config.num_harmonics,
                length=len(y),
                periods=self.config.multi_harmonic_config.periods
            )[:, 1:]

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


        return self

    def predict(self, X):
        check_is_fitted(self, ['problem_'])
        X = check_array(X, ensure_min_features=len(self.config.exog_config or []))



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
    y = np.log(df.loc["2020":"2021"]["RT_Demand"]).values
    x_temp = df.loc["2020":"2021"]["Dry_Bulb"].values
    X = x_temp.reshape(-1, 1)  # Shape: (n_samples, 1) - just temperature

    print(f"Data loaded: {len(y)} samples")
    print(f"Temperature range: [{np.min(x_temp):.2f}, {np.max(x_temp):.2f}]")

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
