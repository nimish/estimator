import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy import stats

    from tsgam_estimator import (
        TsgamEstimator,
        TsgamEstimatorConfig,
        TsgamMultiPeriodicConfig,
        TsgamSolverConfig,
        TsgamSplineConfig,
    )
    from example_tidal import (
        STATION_CATALOG,
        TIDE_TO_WEATHER,
        load_lcd_weather,
        load_station,
        merge_tidal_weather,
        download_lcd_weather,
        TIDAL_CONSTITUENT_PERIODS_HOURS as PERIODS,
    )
    from tidal_analysis_helpers import compute_lagged_correlation, infer_samples_per_hour
    from tidal_model_shared import (
        prepare_split_regressors,
        tidal_metrics,
    )

    return (
        PERIODS,
        STATION_CATALOG,
        TIDE_TO_WEATHER,
        TsgamEstimator,
        TsgamEstimatorConfig,
        TsgamMultiPeriodicConfig,
        TsgamSolverConfig,
        TsgamSplineConfig,
        compute_lagged_correlation,
        download_lcd_weather,
        go,
        infer_samples_per_hour,
        load_lcd_weather,
        load_station,
        make_subplots,
        merge_tidal_weather,
        mo,
        np,
        pd,
        prepare_split_regressors,
        stats,
        tidal_metrics,
    )


@app.cell
def _(STATION_CATALOG, mo):
    _options = {v["name"]: k for k, v in STATION_CATALOG.items()}
    station_picker = mo.ui.dropdown(options=_options, value="The Battery, NY", label="Station")
    mo.vstack([station_picker], justify="start")
    return (station_picker,)


@app.cell
def _(TIDE_TO_WEATHER, mo, station_picker):
    mo.stop(not station_picker.value, mo.md("*Select a station to enable weather options.*"))
    _sid = station_picker.value
    _wx = TIDE_TO_WEATHER.get(_sid)

    toggles = []
    if _wx:
        _wx_id, _wx_name = _wx
        use_weather = mo.ui.switch(value=True, label=f"Merge LCD weather from {_wx_name}")
        download_toggle = mo.ui.switch(label="Download weather data if not found", value=True)

        toggles = [use_weather, download_toggle]
    else:
        use_weather = mo.ui.switch(value=False, label="No mapped weather station", disabled=True)
        toggles = [use_weather]

    mo.hstack(toggles, justify="start")
    return download_toggle, use_weather


@app.cell
def _(
    TIDE_TO_WEATHER,
    download_lcd_weather,
    download_toggle,
    infer_samples_per_hour,
    load_lcd_weather,
    load_station,
    merge_tidal_weather,
    mo,
    station_picker,
    use_weather,
):

    df = load_station(station_picker.value)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    _sid = station_picker.value
    _wx = TIDE_TO_WEATHER.get(_sid)
    weather_status = ""

    if use_weather.value and _wx:
        _wx_id, _wx_name = _wx

        try:
            _wdf = load_lcd_weather(
                "data/tidal", station_id=_wx_id,
                begin_date=str(df.index[0].date()),
                end_date=str(df.index[-1].date()),
            )
        except FileNotFoundError:
            if not download_toggle.value:
                raise
            download_lcd_weather(
                "data/tidal", station_id=_wx_id,
            )
            _wdf = load_lcd_weather(
                "data/tidal", station_id=_wx_id,
                begin_date=str(df.index[0].date()),
                end_date=str(df.index[-1].date()),
            )

        df = merge_tidal_weather(df, _wdf)
        weather_status = f"Merged {len(_wdf.columns)} weather columns from {_wx_name}"

    if "pressure" in df.columns:
        df["dp_dt"] = df["pressure"].diff()
    if "wind_u" in df.columns and "wind_v" in df.columns:
        df["wind_stress"] = df["wind_u"] ** 2 + df["wind_v"] ** 2

    sph = infer_samples_per_hour(df.index)
    date_min = df.index[0].date()
    date_max = df.index[-1].date()
    if weather_status:
        mo.output.replace(mo.md(f"*{weather_status}*"))
    return date_max, date_min, df, sph


@app.cell
def _(date_max, date_min, mo):
    explore_start = mo.ui.date(value=date_min, label="From")
    explore_end = mo.ui.date(value=date_max, label="To")
    explore_resample = mo.ui.dropdown(
        options={"---": None, "Hourly": "1h", "6-hourly": "6h", "Daily": "1D"},
        value="---", label="Resample",
    )
    mo.hstack([explore_start, explore_end, explore_resample], justify="start")
    return explore_end, explore_resample, explore_start


@app.cell
def _(COLUMN_LABELS, df, explore_end, explore_resample, explore_start, go, make_subplots):
    _w = df[str(explore_start.value) : str(explore_end.value)]
    if explore_resample.value is not None:
        _w = _w.resample(explore_resample.value).mean()

    _cols = [c for c in COLUMN_LABELS if c in _w.columns and _w[c].notna().any()]

    _fig = make_subplots(rows=len(_cols), cols=1, shared_xaxes=True,
                         vertical_spacing=0.04)
    for _i, _col in enumerate(_cols, 1):
        _fig.add_trace(
            go.Scattergl(x=_w.index, y=_w[_col], mode="lines",
                         line=dict(width=0.8), name=_col),
            row=_i, col=1,
        )
        _fig.update_yaxes(title_text=COLUMN_LABELS.get(_col, _col), row=_i, col=1)

    _n = len(_w.dropna(how="all"))
    _suffix = f", {explore_resample.value}" if explore_resample.value else ""
    _fig.update_layout(
        height=180 * len(_cols),
        title=f"{_n:,} points ({explore_start.value} \u2014 {explore_end.value}{_suffix})",
        showlegend=False, margin=dict(l=60, r=20, t=40, b=30),
    )
    _fig
    return


@app.cell
def _(df, go, mo, np):
    _candidates = ["water_level", "pressure", "dp_dt", "water_temp", "wind_u", "wind_v",
                    "air_temp", "wind_speed", "wind_stress", "lcd_slp"]
    _cols = [c for c in _candidates if c in df.columns and df[c].notna().any()]
    mo.stop(len(_cols) < 2, mo.md("*Merge weather data above to see regressor correlations.*"))

    _corr = df[_cols].corr()
    _fig = go.Figure(go.Heatmap(
        z=_corr.values, x=_corr.columns.tolist(), y=_corr.columns.tolist(),
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(_corr.values, 2), texttemplate="%{text}",
    ))
    _fig.update_layout(
        title="Pairwise Pearson correlation",
        width=520, height=520,
        margin=dict(l=100, r=20, t=40, b=100),
    )
    _fig
    return


@app.cell
def _(
    PERIODS,
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiPeriodicConfig,
    TsgamSolverConfig,
    TsgamSplineConfig,
    np,
    pd,
    prepare_split_regressors,
    tidal_metrics,
):
    def run_tidal_model(
        component_mask: dict[str, bool],
        *,
        df: pd.DataFrame,
        sph: int,
        harmonic_orders: dict[str, int],
        lag_ranges: dict[str, tuple[int, int]],
        train_start: str,
        train_end: str,
        test_end: str,
    ) -> dict:
        """Fit model with a subset of components, return full results."""
        _split = pd.Timestamp(train_end)
        _dfw = df[train_start:test_end]
        _tr = _dfw[_dfw.index < _split]
        _te = _dfw[_dfw.index >= _split]
        _ok_tr = _tr["water_level"].notna()
        _y_tr = _tr.loc[_ok_tr, "water_level"].values
        _ok_te = _te["water_level"].notna()

        picked = {}
        for name, active in component_mask.items():
            if active and name in PERIODS:
                order = harmonic_orders.get(name, 0)
                if order > 0:
                    picked[name] = (PERIODS[name], order)

        _reg_names = [c for c, on in component_mask.items() if on and c not in PERIODS]

        _multi_periodic = None
        if picked:
            _multi_periodic = TsgamMultiPeriodicConfig(
                periods=[p * sph for p, _ in picked.values()],
                num_harmonics=[n for _, n in picked.values()],
                reg_weight=1e-4,
            )

        _exog_config = None
        _X_tr_fit = pd.DataFrame(index=_tr.index[_ok_tr])
        _X_tr_pred = pd.DataFrame(index=_tr.index)
        _X_te_pred = pd.DataFrame(index=_te.index)
        _active_regs: list[str] = []
        if _reg_names:
            _X_tr_r, _X_te_r, _active_regs, _ = prepare_split_regressors(
                _tr, _te, _reg_names,
            )
            if _active_regs:
                _exog_config = []
                for _col in _active_regs:
                    _lo, _hi = lag_ranges.get(_col, (-2, 0))
                    _exog_config.append(TsgamSplineConfig(
                        n_knots=10 if _col == "pressure" else 8,
                        lags=[h * sph for h in range(_lo, _hi + 1)],
                        reg_weight=1e-5,
                        diff_reg_weight=0.3,
                    ))
                _X_tr_fit = _X_tr_r.loc[_ok_tr]
                _X_tr_pred = _X_tr_r
                _X_te_pred = _X_te_r

        def _pack(yh_tr, yh_te):
            _te_obs = _te["water_level"].values
            return {
                "metrics_train": tidal_metrics(_y_tr, yh_tr[_ok_tr]),
                "metrics_test": tidal_metrics(
                    _te.loc[_ok_te, "water_level"].values, yh_te[_ok_te],
                ),
                "te_index": _te.index,
                "te_obs": _te_obs,
                "te_pred": yh_te,
                "te_obs_clean": _te.loc[_ok_te, "water_level"].values,
                "te_pred_clean": yh_te[_ok_te],
                "residuals": _te_obs - yh_te,
                "picked": picked,
                "active_regs": _active_regs,
                "n_train": len(_y_tr),
                "n_test": int(_ok_te.sum()),
                "sph": sph,
            }

        if _multi_periodic is None and _exog_config is None:
            _mean = np.nanmean(_y_tr)
            return _pack(np.full(len(_tr), _mean), np.full(len(_te), _mean))

        try:
            _mdl = TsgamEstimator(TsgamEstimatorConfig(
                multi_periodic_config=_multi_periodic,
                exog_config=_exog_config,
                solver_config=TsgamSolverConfig(solver="SCS", verbose=False),
            ))
            _mdl.fit(_X_tr_fit, _y_tr)
            return _pack(_mdl.predict(_X_tr_pred), _mdl.predict(_X_te_pred))
        except Exception:
            return _pack(np.full(len(_tr), np.nan), np.full(len(_te), np.nan))

    return (run_tidal_model,)


@app.cell
def _():
    COLUMN_LABELS = {
        "water_level": "Water level (m)",
        "pressure": "Pressure (hPa)",
        "dp_dt": "dP/dt (hPa/step)",
        "water_temp": "Water temp (\u00b0C)",
        "air_temp": "Air temp (\u00b0C)",
        "wind_u": "Wind U (m/s)",
        "wind_v": "Wind V (m/s)",
        "wind_speed": "Wind speed (m/s)",
        "wind_stress": "Wind stress (m\u00b2/s\u00b2)",
        "lcd_slp": "LCD sea level pressure (hPa)",
    }
    return (COLUMN_LABELS,)


@app.cell
def _():
    def collect_model_params(harmonic_inputs, regressor_lags, regressor_toggles,
                             df, sph, train_start, train_end, test_end):
        """Read current widget values into a component mask and model kwargs dict."""
        mask = {k: int(w.value) > 0 for k, w in harmonic_inputs.items()}
        mask.update({c: sw.value for c, sw in regressor_toggles.items()})
        kw = dict(
            df=df, sph=sph,
            harmonic_orders={k: int(w.value) for k, w in harmonic_inputs.items()},
            lag_ranges={c: regressor_lags[c].value for c in regressor_lags},
            train_start=str(train_start.value),
            train_end=str(train_end.value),
            test_end=str(test_end.value),
        )
        return mask, kw

    return (collect_model_params,)


@app.cell
def _(PERIODS, date_max, date_min, df, mo, pd):
    _defaults = {
        "M2": 4, "S2": 1, "N2": 1, "K1": 2,
        "O1": 1, "Mf": 1, "Mm": 1, "annual": 2,
    }
    harmonic_inputs = {
        name: mo.ui.slider(
            start=0, stop=8, value=n,
            label=f"{name} ({PERIODS.get(name, 8766.0):.1f} h)",
            show_value=True, full_width=True,
        )
        for name, n in _defaults.items()
    }

    _met_candidates = ["pressure", "dp_dt", "water_temp", "wind_u", "wind_v",
                        "air_temp", "wind_stress"]
    _available = [c for c in _met_candidates if c in df.columns and df[c].notna().any()]

    _lag_defaults = {
        "pressure": (-2, 0), "dp_dt": (-2, 0), "water_temp": (0, 0),
        "wind_u": (-1, 0), "wind_v": (-1, 0), "air_temp": (0, 0),
        "wind_stress": (-1, 0),
    }
    regressor_toggles = {}
    regressor_lags = {}
    _reg_rows = []
    for c in _available:
        regressor_toggles[c] = mo.ui.switch(value=False, label=c)
        _lo, _hi = _lag_defaults.get(c, (-2, 0))
        regressor_lags[c] = mo.ui.range_slider(
            start=-6, stop=6, value=(_lo, _hi),
            label="lag (h)", show_value=True,
        )
        _reg_rows.append(mo.hstack([regressor_toggles[c], regressor_lags[c]], justify="start"))

    train_start = mo.ui.date(value=date_min, label="Train start")
    train_end = mo.ui.date(value=pd.Timestamp("2024-01-01").date(), label="Train end")
    test_end = mo.ui.date(value=date_max, label="Test end")
    run_fit = mo.ui.run_button(label="Fit model")

    mo.vstack([
        mo.md("## Configure model"),
        mo.hstack([
            mo.vstack([mo.md("**Harmonics** (0 = exclude)")] + list(harmonic_inputs.values())),
            mo.vstack([mo.md("**Regressors** (toggle + lag range in hours)")] + _reg_rows)
                if _available else mo.md(""),
            mo.vstack([mo.md("**Date ranges**"), train_start, train_end, test_end]),
        ], justify="start", gap=2),
        run_fit,
    ])
    return (
        harmonic_inputs,
        regressor_lags,
        regressor_toggles,
        run_fit,
        test_end,
        train_end,
        train_start,
    )


@app.cell
def _(
    collect_model_params,
    df,
    harmonic_inputs,
    mo,
    pd,
    regressor_lags,
    regressor_toggles,
    run_fit,
    run_tidal_model,
    sph,
    test_end,
    train_end,
    train_start,
):
    mo.stop(not run_fit.value, mo.md("*Configure parameters above and click **Fit model**.*"))

    _mask, _kw = collect_model_params(
        harmonic_inputs, regressor_lags, regressor_toggles,
        df, sph, train_start, train_end, test_end,
    )
    mo.stop(not any(_mask.values()),
            mo.md("*Set at least one constituent to harmonics > 0.*"))

    fit_result = run_tidal_model(_mask, **_kw)

    _parts = [", ".join(fit_result["picked"])]
    if fit_result["active_regs"]:
        _parts.append(f"+ {', '.join(fit_result['active_regs'])}")
    fit_label = (
        f"{' '.join(_parts)} \u2014 "
        f"{fit_result['n_train']:,} train / {fit_result['n_test']:,} test"
    )

    mo.Html(pd.DataFrame({
        "train": fit_result["metrics_train"],
        "test": fit_result["metrics_test"],
    }).T.to_html())
    return fit_label, fit_result


@app.cell
def _(fit_label, fit_result, go, make_subplots):
    _idx = fit_result["te_index"]
    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         vertical_spacing=0.06, row_heights=[0.7, 0.3])
    _fig.add_trace(
        go.Scattergl(x=_idx, y=fit_result["te_obs"], mode="lines",
                     line=dict(width=0.8, color="steelblue"), name="observed",
                     opacity=0.8),
        row=1, col=1,
    )
    _fig.add_trace(
        go.Scattergl(x=_idx, y=fit_result["te_pred"], mode="lines",
                     line=dict(width=0.8, color="coral"), name="predicted",
                     opacity=0.8),
        row=1, col=1,
    )
    _fig.add_trace(
        go.Scattergl(x=_idx, y=fit_result["residuals"], mode="lines",
                     line=dict(width=0.6, color="seagreen"), name="residual",
                     showlegend=False),
        row=2, col=1,
    )
    _fig.add_hline(y=0, line_width=0.5, line_color="black", row=2, col=1)
    _fig.update_yaxes(title_text="Water level (m)", row=1, col=1)
    _fig.update_yaxes(title_text="Residual (m)", row=2, col=1)
    _fig.update_layout(
        height=500, title=fit_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=20, t=60, b=30),
    )
    _fig
    return


@app.cell
def _(compute_lagged_correlation, df, fit_result, go, mo, np, pd, stats):
    _resid = fit_result["residuals"]
    _idx = fit_result["te_index"]
    _obs = fit_result["te_obs_clean"]
    _pred = fit_result["te_pred_clean"]
    _sph = fit_result["sph"]
    _clean = _resid[np.isfinite(_resid)]

    # --- Predicted vs Observed ---
    _lo, _hi = min(_pred.min(), _obs.min()), max(_pred.max(), _obs.max())
    _scatter = go.Figure()
    _scatter.add_trace(go.Scattergl(
        x=_pred, y=_obs, mode="markers",
        marker=dict(size=2, opacity=0.15, color="steelblue"), showlegend=False,
    ))
    _scatter.add_trace(go.Scatter(
        x=[_lo, _hi], y=[_lo, _hi], mode="lines",
        line=dict(dash="dash", width=0.8, color="black"), showlegend=False,
    ))
    _scatter.update_layout(
        width=450, height=450, title="Predicted vs Observed",
        xaxis=dict(title="Predicted (m)", scaleanchor="y", range=[_lo, _hi]),
        yaxis=dict(title="Observed (m)", range=[_lo, _hi]),
        margin=dict(l=60, r=20, t=40, b=50),
    )

    # --- Q-Q plot ---
    ((_theo, _sample), (_slope, _intercept, _)) = stats.probplot(_clean, dist="norm")
    _qq = go.Figure()
    _qq.add_trace(go.Scattergl(
        x=_theo, y=_sample, mode="markers",
        marker=dict(size=3, opacity=0.3, color="steelblue"), name="residuals",
    ))
    _qq.add_trace(go.Scatter(
        x=_theo, y=_slope * _theo + _intercept, mode="lines",
        line=dict(color="red", width=1, dash="dash"), name="normal ref",
    ))
    _qq.update_layout(
        width=450, height=450, title="Q-Q plot (residuals)",
        xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles (m)",
        margin=dict(l=60, r=20, t=40, b=50),
    )

    # --- Residual histogram ---
    _mu, _sigma = _clean.mean(), _clean.std()
    _xr = np.linspace(_clean.min(), _clean.max(), 200)
    _bw = (_clean.max() - _clean.min()) / 80
    _hist = go.Figure()
    _hist.add_trace(go.Histogram(
        x=_clean, nbinsx=80, name="residuals",
        marker_color="steelblue", opacity=0.7,
    ))
    _hist.add_trace(go.Scatter(
        x=_xr, y=stats.norm.pdf(_xr, _mu, _sigma) * len(_clean) * _bw,
        mode="lines", line=dict(color="red", width=1.5),
        name=f"N({_mu:.3f}, {_sigma:.3f}\u00b2)",
    ))
    _hist.update_layout(
        title="Residual distribution",
        xaxis_title="Residual (m)", yaxis_title="Count",
        margin=dict(l=60, r=20, t=40, b=50),
        height=350, barmode="overlay",
    )

    # --- Rolling RMSE ---
    _window = int(168 * _sph)
    _rolling = pd.Series(_resid ** 2, index=_idx).rolling(
        _window, min_periods=_window // 2,
    ).mean().pipe(np.sqrt)
    _rmse_fig = go.Figure()
    _rmse_fig.add_trace(go.Scattergl(
        x=_rolling.index, y=_rolling.values, mode="lines",
        line=dict(width=0.8, color="steelblue"), showlegend=False,
    ))
    _rmse_fig.update_layout(
        title="Rolling RMSE (7-day window)",
        yaxis_title="RMSE (m)", height=300,
        margin=dict(l=60, r=20, t=40, b=30),
    )

    # --- Residual-regressor cross-correlation ---
    _reg_candidates = ["pressure", "dp_dt", "water_temp", "wind_u", "wind_v",
                        "air_temp", "wind_speed", "wind_stress", "lcd_slp"]
    _regs = [c for c in _reg_candidates if c in df.columns and df[c].notna().any()]
    _xcorr = None
    if _regs:
        _ok = np.isfinite(_resid)
        _te_df = df.loc[_idx[_ok]]
        _max_lag = int(12 * _sph)
        _xcorr = go.Figure()
        for _col in _regs:
            _lc = compute_lagged_correlation(_resid[_ok], _te_df[_col].values, _max_lag)
            _lc["lag_hours"] = _lc["lag"] / _sph
            _xcorr.add_trace(go.Scatter(
                x=_lc["lag_hours"], y=_lc["correlation"],
                mode="lines", name=_col, line=dict(width=1.5),
            ))
        _xcorr.add_hline(y=0, line_width=0.5, line_color="black")
        _xcorr.add_vline(x=0, line_width=2.0, line_color="red")
        _xcorr.update_layout(
            title="Residual\u2013regressor cross-correlation (\u00b112 h)",
            xaxis_title="Lag (hours)", yaxis_title="Pearson r",
            height=400, margin=dict(l=60, r=20, t=40, b=50),
        )

    _tabs = {
        "Pred vs Obs": _scatter,
        "Q-Q": _qq,
        "Residual dist": _hist,
        "Rolling RMSE": _rmse_fig,
    }
    if _xcorr is not None:
        _tabs["Resid \u00d7 regressor"] = _xcorr

    mo.ui.tabs(_tabs, lazy=False)
    return


@app.cell
def _(np):
    from math import factorial

    def compute_shapley(
        results: dict[int, dict],
        components: list[str],
        metric: str,
        baseline: float,
    ) -> dict[str, float]:
        """Shapley attribution over precomputed coalition metrics."""
        n = len(components)
        vals = {0: baseline}
        for bits, m in results.items():
            v = m.get(metric, baseline)
            vals[bits] = v if np.isfinite(v) else baseline
        shapley = {}
        for i, comp in enumerate(components):
            sv = 0.0
            for S in range(2**n):
                if S & (1 << i):
                    continue
                s_size = bin(S).count("1")
                w = factorial(s_size) * factorial(n - s_size - 1) / factorial(n)
                sv += w * (vals[S | (1 << i)] - vals[S])
            shapley[comp] = sv
        return shapley

    return (compute_shapley,)


@app.cell
def _(mo):
    run_shapley = mo.ui.run_button(label="Run Shapley analysis")
    mo.vstack([mo.md("## Shapley value analysis"), run_shapley])
    return (run_shapley,)


@app.cell
def _(
    collect_model_params,
    compute_shapley,
    df,
    go,
    harmonic_inputs,
    make_subplots,
    mo,
    np,
    regressor_lags,
    regressor_toggles,
    run_shapley,
    run_tidal_model,
    sph,
    test_end,
    train_end,
    train_start,
):
    mo.stop(not run_shapley.value,
            mo.md("*Click above to compute Shapley values for active components.*"))
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from functools import partial

    _mask, _kw = collect_model_params(
        harmonic_inputs, regressor_lags, regressor_toggles,
        df, sph, train_start, train_end, test_end,
    )
    _components = [c for c, on in _mask.items() if on]
    _n = len(_components)

    mo.stop(_n < 2, mo.md("*Need at least 2 active components for Shapley analysis.*"))
    mo.stop(_n > 12, mo.md(f"*{_n} components \u2192 {2**_n:,} model runs \u2014 cap at 12.*"))

    _run = partial(run_tidal_model, **_kw)

    _total = 2**_n - 1
    _failed = 0
    _workers = min(os.cpu_count() or 4, _total, 8)
    _results: dict[int, dict] = {}

    with mo.status.progress_bar(total=_total) as _bar:
        _masks = {
            bits: {c: bool(bits & (1 << i)) for i, c in enumerate(_components)}
            for bits in range(1, 2**_n)
        }
        with ThreadPoolExecutor(max_workers=_workers) as _pool:
            _futures = {
                _pool.submit(_run, mask): bits
                for bits, mask in _masks.items()
            }
            for _fut in as_completed(_futures):
                _bits = _futures[_fut]
                _m = _fut.result()["metrics_test"]
                if np.isnan(_m.get("r2", 0.0)):
                    _failed += 1
                _results[_bits] = _m
                _bar.update()

    _baseline_rmse = _run({c: False for c in _components})["metrics_test"]["rmse"]
    _shap_r2 = compute_shapley(_results, _components, "r2", 0.0)
    _shap_rmse = compute_shapley(_results, _components, "rmse", _baseline_rmse)

    _full = 2**_n - 1
    _full_r2 = _results[_full].get("r2", 0.0) if np.isfinite(_results[_full].get("r2", 0.0)) else 0.0
    _full_rmse = _results[_full].get("rmse", _baseline_rmse)

    _fig = make_subplots(rows=1, cols=2, subplot_titles=["R\u00b2 attribution", "RMSE attribution"])
    for _col_idx, (_shap, _bl, _full_v, _ylabel) in enumerate([
        (_shap_r2, 0.0, _full_r2, "R\u00b2"),
        (_shap_rmse, _baseline_rmse, _full_rmse, "RMSE (m)"),
    ], 1):
        _sorted = sorted(_shap.items(), key=lambda x: abs(x[1]), reverse=True)
        _names = [x[0] for x in _sorted]
        _vals = [x[1] for x in _sorted]
        _fig.add_trace(go.Waterfall(
            x=["baseline"] + _names + ["full model"],
            y=[_bl] + _vals + [0.0],
            measure=["absolute"] + ["relative"] * _n + ["total"],
            text=[f"{v:.4f}" for v in [_bl] + _vals + [_full_v]],
            textposition="outside",
            increasing=dict(marker_color="steelblue" if _col_idx == 1 else "coral"),
            decreasing=dict(marker_color="coral" if _col_idx == 1 else "steelblue"),
            totals=dict(marker_color="midnightblue"),
            connector=dict(line=dict(color="gray", width=0.5, dash="dot")),
        ), row=1, col=_col_idx)
        _fig.update_yaxes(title_text=_ylabel, row=1, col=_col_idx)

    _title = f"Shapley component attribution ({_total} coalitions)"
    if _failed:
        _title += f", {_failed} solver failures"
    _fig.update_layout(
        title=_title, showlegend=False,
        height=max(400, 40 * _n),
        margin=dict(l=60, r=40, t=60, b=80),
    )
    _fig.update_xaxes(tickangle=-30)
    _fig
    return


if __name__ == "__main__":
    app.run()
