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
    from tidal_analysis_helpers import compute_lagged_correlation, compute_periodogram, infer_samples_per_hour
    from tidal_model_shared import (
        make_tidal_spline_configs,
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
def _(df):
    viz_df = df.copy()
    viz_df['time'] = viz_df.index
    viz_df
    return


@app.cell
def _(df, explore_end, explore_resample, explore_start, go, make_subplots):
    _w = df[str(explore_start.value) : str(explore_end.value)]
    if explore_resample.value is not None:
        _w = _w.resample(explore_resample.value).mean()

    _ylabels = {
        "water_level": "Water level (m)",
        "pressure": "Pressure (hPa)",
        "dp_dt": "dP/dt (hPa/step)",
        "water_temp": "Water temp (°C)",
        "air_temp": "Air temp (°C)",
        "wind_u": "Wind U (m/s)",
        "wind_v": "Wind V (m/s)",
        "wind_speed": "Wind speed (m/s)",
        "wind_stress": "Wind stress (m²/s²)",
        "lcd_slp": "LCD sea level pressure (hPa)",
    }
    _cols = [c for c in _ylabels if c in _w.columns and _w[c].notna().any()]

    _fig = make_subplots(rows=len(_cols), cols=1, shared_xaxes=True,
                         vertical_spacing=0.04)
    for _i, _col in enumerate(_cols, 1):
        _fig.add_trace(
            go.Scattergl(x=_w.index, y=_w[_col], mode="lines",
                         line=dict(width=0.8), name=_col),
            row=_i, col=1,
        )
        _fig.update_yaxes(title_text=_ylabels.get(_col, _col), row=_i, col=1)

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
    PERIODS,
    TsgamEstimator,
    TsgamEstimatorConfig,
    TsgamMultiPeriodicConfig,
    TsgamSolverConfig,
    TsgamSplineConfig,
    df,
    harmonic_inputs,
    mo,
    pd,
    prepare_split_regressors,
    regressor_lags,
    regressor_toggles,
    run_fit,
    sph,
    test_end,
    tidal_metrics,
    train_end,
    train_start,
):
    mo.stop(not run_fit.value, mo.md("*Configure parameters above and click **Fit model**.*"))

    picked = {k: (PERIODS.get(k, 8766.0), int(w.value))
              for k, w in harmonic_inputs.items() if int(w.value) > 0}
    mo.stop(not picked, mo.md("*Set at least one constituent to harmonics > 0.*"))

    _split = pd.Timestamp(train_end.value)
    _dfw = df[str(train_start.value) : str(test_end.value)]
    _tr, _te = _dfw[_dfw.index < _split], _dfw[_dfw.index >= _split]

    _ok_tr = _tr["water_level"].notna()
    _y_tr = _tr.loc[_ok_tr, "water_level"].values

    _selected_regs = [c for c, sw in regressor_toggles.items() if sw.value]
    if _selected_regs:
        _X_tr, _X_te, _active, _dropped = prepare_split_regressors(
            _tr, _te, _selected_regs,
        )
        _exog_config = []
        for _col in _active:
            _lo, _hi = regressor_lags[_col].value
            _exog_config.append(TsgamSplineConfig(
                n_knots=10 if _col == "pressure" else 8,
                lags=[h * sph for h in range(_lo, _hi + 1)],
                reg_weight=1e-5,
                diff_reg_weight=0.3,
            ))
        _X_tr_fit = _X_tr.loc[_ok_tr]
    else:
        _X_tr_fit = pd.DataFrame(index=_tr.index[_ok_tr])
        _X_te = pd.DataFrame(index=_te.index)
        _exog_config = None
        _active = []

    _mdl = TsgamEstimator(TsgamEstimatorConfig(
        multi_periodic_config=TsgamMultiPeriodicConfig(
            periods=[p * sph for p, _ in picked.values()],
            num_harmonics=[n for _, n in picked.values()],
            reg_weight=1e-4,
        ),
        exog_config=_exog_config or None,
        solver_config=TsgamSolverConfig(solver="SCS", verbose=False),
    ))
    _mdl.fit(_X_tr_fit, _y_tr)

    _yh_tr = _mdl.predict(_X_tr if _selected_regs else pd.DataFrame(index=_tr.index))
    _yh_te = _mdl.predict(_X_te)
    _ok_te = _te["water_level"].notna()

    fit_metrics = pd.DataFrame({
        "train": tidal_metrics(_y_tr, _yh_tr[_ok_tr]),
        "test": tidal_metrics(_te.loc[_ok_te, "water_level"].values, _yh_te[_ok_te]),
    }).T
    _parts = [', '.join(picked)]
    if _active:
        _parts.append(f"+ {', '.join(_active)}")
    fit_label = f"{' '.join(_parts)} \u2014 {len(_y_tr):,} train / {_ok_te.sum():,} test"
    fit_te_index = _te.index
    fit_te_obs = _te["water_level"].values
    fit_te_pred = _yh_te
    fit_te_obs_clean = _te.loc[_ok_te, "water_level"].values
    fit_te_pred_clean = _yh_te[_ok_te]
    fit_residuals = fit_te_obs - fit_te_pred
    fit_sph = sph

    mo.Html(fit_metrics.to_html())
    return (
        fit_label,
        fit_residuals,
        fit_sph,
        fit_te_index,
        fit_te_obs,
        fit_te_obs_clean,
        fit_te_pred,
        fit_te_pred_clean,
    )


@app.cell
def _(
    fit_label,
    fit_residuals,
    fit_te_index,
    fit_te_obs,
    fit_te_pred,
    go,
    make_subplots,
):
    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         vertical_spacing=0.06, row_heights=[0.7, 0.3])
    _fig.add_trace(
        go.Scattergl(x=fit_te_index, y=fit_te_obs, mode="lines",
                     line=dict(width=0.8, color="steelblue"), name="observed",
                     opacity=0.8),
        row=1, col=1,
    )
    _fig.add_trace(
        go.Scattergl(x=fit_te_index, y=fit_te_pred, mode="lines",
                     line=dict(width=0.8, color="coral"), name="predicted",
                     opacity=0.8),
        row=1, col=1,
    )
    _fig.add_trace(
        go.Scattergl(x=fit_te_index, y=fit_residuals, mode="lines",
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
def _():
    return


@app.cell
def _(fit_te_obs_clean, fit_te_pred_clean, go):
    _lo = min(fit_te_pred_clean.min(), fit_te_obs_clean.min())
    _hi = max(fit_te_pred_clean.max(), fit_te_obs_clean.max())
    _fig = go.Figure()
    _fig.add_trace(go.Scattergl(
        x=fit_te_pred_clean, y=fit_te_obs_clean, mode="markers",
        marker=dict(size=2, opacity=0.15, color="steelblue"),
        showlegend=False,
    ))
    _fig.add_trace(go.Scatter(
        x=[_lo, _hi], y=[_lo, _hi], mode="lines",
        line=dict(dash="dash", width=0.8, color="black"),
        showlegend=False,
    ))
    _fig.update_layout(
        width=450, height=450,
        title="Predicted vs Observed",
        xaxis=dict(title="Predicted (m)", scaleanchor="y", range=[_lo, _hi]),
        yaxis=dict(title="Observed (m)", range=[_lo, _hi]),
        margin=dict(l=60, r=20, t=40, b=50),
    )
    _fig
    return


@app.cell
def _(fit_residuals, go, np, stats):
    _clean = fit_residuals[np.isfinite(fit_residuals)]
    ((_theo, _sample), (_slope, _intercept, _)) = stats.probplot(_clean, dist="norm")
    _fit_line_y = _slope * _theo + _intercept

    _fig = go.Figure()
    _fig.add_trace(go.Scattergl(
        x=_theo, y=_sample, mode="markers",
        marker=dict(size=3, opacity=0.3, color="steelblue"),
        name="residuals",
    ))
    _fig.add_trace(go.Scatter(
        x=_theo, y=_fit_line_y, mode="lines",
        line=dict(color="red", width=1, dash="dash"),
        name="normal ref",
    ))
    _fig.update_layout(
        width=450, height=450,
        title="Q-Q plot (residuals)",
        xaxis_title="Theoretical quantiles",
        yaxis_title="Sample quantiles (m)",
        margin=dict(l=60, r=20, t=40, b=50),
    )
    _fig
    return


@app.cell
def _(fit_residuals, go, np, stats):
    _clean = fit_residuals[np.isfinite(fit_residuals)]
    _mu, _sigma = _clean.mean(), _clean.std()
    _x = np.linspace(_clean.min(), _clean.max(), 200)
    _pdf = stats.norm.pdf(_x, _mu, _sigma)
    _bin_width = (_clean.max() - _clean.min()) / 80
    _pdf_scaled = _pdf * len(_clean) * _bin_width

    _fig = go.Figure()
    _fig.add_trace(go.Histogram(
        x=_clean, nbinsx=80, name="residuals",
        marker_color="steelblue", opacity=0.7,
    ))
    _fig.add_trace(go.Scatter(
        x=_x, y=_pdf_scaled, mode="lines",
        line=dict(color="red", width=1.5),
        name=f"N({_mu:.3f}, {_sigma:.3f}\u00b2)",
    ))
    _fig.update_layout(
        title="Residual distribution",
        xaxis_title="Residual (m)", yaxis_title="Count",
        margin=dict(l=60, r=20, t=40, b=50),
        height=350, barmode="overlay",
    )
    _fig
    return


@app.cell
def _(fit_residuals, fit_sph, fit_te_index, go, np, pd):
    _window = int(168 * fit_sph)
    _rmse = pd.Series(fit_residuals ** 2, index=fit_te_index).rolling(
        _window, min_periods=_window // 2,
    ).mean().pipe(np.sqrt)

    _fig = go.Figure()
    _fig.add_trace(go.Scattergl(
        x=_rmse.index, y=_rmse.values, mode="lines",
        line=dict(width=0.8, color="steelblue"), showlegend=False,
    ))
    _fig.update_layout(
        title="Rolling RMSE (7-day window)",
        xaxis_title=None, yaxis_title="RMSE (m)",
        margin=dict(l=60, r=20, t=40, b=30),
        height=300,
    )
    _fig
    return


@app.cell
def _(
    compute_lagged_correlation,
    df,
    fit_residuals,
    fit_sph,
    fit_te_index,
    go,
    mo,
    np,
):
    _candidates = ["pressure", "dp_dt", "water_temp", "wind_u", "wind_v", "air_temp",
                    "wind_speed", "wind_stress", "lcd_slp"]
    _regs = [c for c in _candidates if c in df.columns and df[c].notna().any()]
    mo.stop(not _regs, mo.md("*No regressors available for residual analysis.*"))

    _clean = np.isfinite(fit_residuals)
    _resid = fit_residuals[_clean]
    _te_df = df.loc[fit_te_index[_clean]]
    _max_lag = int(12 * fit_sph)

    _fig = go.Figure()
    for _col in _regs:
        _feat = _te_df[_col].values
        _lc = compute_lagged_correlation(_resid, _feat, _max_lag)
        _lc["lag_hours"] = _lc["lag"] / fit_sph
        _fig.add_trace(go.Scatter(
            x=_lc["lag_hours"], y=_lc["correlation"],
            mode="lines", name=_col, line=dict(width=1.5),
        ))
    _fig.add_hline(y=0, line_width=0.5, line_color="black")
    _fig.update_layout(
        title="Residual\u2013regressor cross-correlation (\u00b112 h)",
        xaxis_title="Lag (hours)",
        yaxis_title="Pearson r",
        height=400,
        margin=dict(l=60, r=20, t=40, b=50),
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
