"""
ETH Datathon 2025 · Axpo Iberia Retail Energy Forecasting
=========================================================

Reference implementation of the day-ahead portfolio forecasting pipeline
described in the team presentation (`slides/presentation.pdf`).

Pipeline (all steps strictly leakage-free w.r.t. the forecast horizon D):

    1. Big-client extraction (training-only statistic)
           Clients whose cumulative active_kw over the training window is in the
           top-K are pulled out of their community and promoted to a synthetic
           one-client "community". This prevents a single dominant consumer
           from dragging the per-client average alpha of a real region.

    2. Structural per-entity feature
           alpha(c, t) = sum_j y_j(c, t) / (n(c, t) * D_hat_Spain(t))
       where D_hat_Spain(t) is REE's Spain day-ahead demand forecast
       (published well before 12:00 on D-1 -> leakage-free at t).

    3. Zero-parameter baseline
           alpha_hat(c, t) = alpha(c, D-7)      # observed 7 days earlier
           y_hat(c, t)     = alpha(c, D-7) * n(c, D-2) * D_hat_Spain(t)

       Every term on the right is available strictly before 12:00 of day D-1,
       so this forecast is producible on time with no look-ahead.

    4. Residual LightGBM per entity
           eps(c, t) = y_true(c, t) - y_hat_baseline(c, t)
       One LightGBM is trained per entity (community or big client). Features
       fed to the model:
           * Weather forecast snapshot available at D-2 (temperature, HDD/CDD,
             radiation, humidity, wind). Never uses observed weather at t.
           * Time / calendar encodings (quarter-hour, day-of-week,
             day-of-month, day-of-year, sin/cos of each).
       The residual target is small, closer to zero-mean, easier to fit than
       the raw level.

    5. Final forecast
           y_hat_final(t) = sum_entities [y_hat_baseline(c, t) + eps_hat(c, t)]

Training & validation
---------------------
We evaluate with a forward-only `sklearn.model_selection.TimeSeriesSplit`.
No row from fold K ever contributes information (features, statistics, or big-
client membership) to the model used to predict fold K -- the split is applied
to the target timestamp and every training-derived statistic (big-client list,
LightGBM fit) is re-computed inside each fold on data strictly earlier than the
validation window.

The Databricks-facing `EnergyConsumptionModel.predict(df, predict_start, predict_end)`
contract is preserved for the scoring job (final cell at bottom of file).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Feature engineering (pandas - same spirit as src/preprocessing.py Spark code)
# ═══════════════════════════════════════════════════════════════════════════

BASE_WEATHER_COLS: Sequence[str] = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "shortwave_radiation",
    "cloud_cover",
    "wind_speed_10m",
)

CALENDAR_COLS: Sequence[str] = (
    "qhour", "dow", "dom", "doy", "month",
    "sin_qhour", "cos_qhour",
    "sin_dow",   "cos_dow",
    "sin_doy",   "cos_doy",
    "is_weekend",
)


def _add_calendar_features(df: pd.DataFrame, ts_col: str = "datetime_local") -> pd.DataFrame:
    ts = df[ts_col]
    out = df.copy()
    out["qhour"]   = (ts.dt.hour * 4 + ts.dt.minute // 15).astype(np.int16)
    out["dow"]     = ts.dt.dayofweek.astype(np.int16)
    out["dom"]     = ts.dt.day.astype(np.int16)
    out["doy"]     = ts.dt.dayofyear.astype(np.int16)
    out["month"]   = ts.dt.month.astype(np.int16)
    out["is_weekend"] = (out["dow"] >= 5).astype(np.int8)

    TAU = 2 * np.pi
    out["sin_qhour"] = np.sin(TAU * out["qhour"] / 96.0)
    out["cos_qhour"] = np.cos(TAU * out["qhour"] / 96.0)
    out["sin_dow"]   = np.sin(TAU * out["dow"] / 7.0)
    out["cos_dow"]   = np.cos(TAU * out["dow"] / 7.0)
    out["sin_doy"]   = np.sin(TAU * out["doy"] / 366.0)
    out["cos_doy"]   = np.cos(TAU * out["doy"] / 366.0)
    return out


def _attach_weather_forecast(
    panel: pd.DataFrame,
    weather_forecast: pd.DataFrame,
    keys: Sequence[str] = ("community_code", "datetime_local"),
) -> pd.DataFrame:
    """
    Merge the D-2 weather forecast onto the panel.

    `weather_forecast` is expected to be the forecast issued on day D-2 for all
    15-minute steps of day D (produced by Open-Meteo's "previous_day_2_runs"
    snapshot in the original Databricks workspace). Using the D-2 run instead
    of the realised weather on D is what keeps the feature leakage-free.
    """
    cols = [c for c in BASE_WEATHER_COLS if c in weather_forecast.columns]
    return panel.merge(
        weather_forecast[list(keys) + cols],
        on=list(keys), how="left",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Big-client extraction (training-only)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BigClientExtractor:
    """
    Promote the top-`k` clients (by cumulative energy on the training window)
    to their own synthetic communities so that per-client averages inside real
    communities are not dominated by a single outlier.

    The list of big clients is fit *only* on the training window passed to
    `.fit()`, never from validation or future data.
    """
    k: int = 30
    big_client_ids: List[str] = field(default_factory=list)

    def fit(self, train_df: pd.DataFrame) -> "BigClientExtractor":
        totals = (
            train_df.groupby("client_id")["active_kw"]
            .sum()
            .sort_values(ascending=False)
        )
        self.big_client_ids = totals.head(self.k).index.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rewrite community_code to `BIG_<client_id>` for every big client."""
        out = df.copy()
        is_big = out["client_id"].isin(self.big_client_ids)
        out.loc[is_big, "community_code"] = "BIG_" + out.loc[is_big, "client_id"].astype(str)
        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)


# ═══════════════════════════════════════════════════════════════════════════
# Per-entity panel (community or big client) and baseline
# ═══════════════════════════════════════════════════════════════════════════

def build_entity_panel(
    consumption_df: pd.DataFrame,
    demand_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate client-level consumption into one row per (community_code,
    datetime_local), then join the Spain demand forecast and compute alpha.

    Expected columns
    ----------------
    consumption_df     : client_id, community_code, datetime_local, active_kw
    demand_forecast    : datetime_local, D_mw
    """
    agg = (
        consumption_df.groupby(["community_code", "datetime_local"], as_index=False)
        .agg(
            total_kw=("active_kw", "sum"),
            n_clients=("client_id", "nunique"),
        )
    )
    agg = agg.merge(demand_forecast, on="datetime_local", how="left")
    # alpha is undefined when either the community is empty or the forecast is 0.
    valid = (agg["n_clients"] > 0) & (agg["D_mw"] > 0) & agg["total_kw"].notna()
    agg["alpha"] = np.where(
        valid,
        agg["total_kw"] / (agg["n_clients"] * agg["D_mw"]),
        np.nan,
    )
    return agg


def attach_baseline(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add the D-7 alpha and D-2 client count snapshots used by the baseline
    forecast:

        y_hat_baseline(c, t) = alpha(c, D-7) * n(c, D-2) * D_mw(t)
    """
    p = panel.sort_values(["community_code", "datetime_local"]).copy()
    p["alpha_D7"] = (
        p.groupby("community_code")["alpha"]
         .shift(7 * 96)   # 96 quarter-hours per day
    )
    p["n_clients_D2"] = (
        p.groupby("community_code")["n_clients"]
         .shift(2 * 96)
    )
    p["y_hat_baseline"] = p["alpha_D7"] * p["n_clients_D2"] * p["D_mw"]
    p["residual"]      = p["total_kw"] - p["y_hat_baseline"]
    return p


# ═══════════════════════════════════════════════════════════════════════════
# Residual LightGBM (one per entity)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ResidualLGBM:
    """
    Fit one `lightgbm.LGBMRegressor` per entity on the baseline residual.

    We deliberately use a tiny model (few leaves, strong L2) because the target
    is already close to zero-mean -- the goal is small, smooth corrections.
    """
    params: dict = field(default_factory=lambda: dict(
        objective="regression_l1",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        lambda_l2=1.0,
        random_state=42,
        verbose=-1,
    ))
    models: dict = field(default_factory=dict)
    feature_cols: List[str] = field(default_factory=list)

    def _build_feature_matrix(self, panel: pd.DataFrame) -> pd.DataFrame:
        weather_cols = [c for c in BASE_WEATHER_COLS if c in panel.columns]
        calendar_cols = [c for c in CALENDAR_COLS if c in panel.columns]
        self.feature_cols = list(dict.fromkeys(weather_cols + calendar_cols))
        return panel[self.feature_cols]

    def fit(self, panel: pd.DataFrame) -> "ResidualLGBM":
        import lightgbm as lgb  # local import so the module can be read on a vanilla Python

        X_all = self._build_feature_matrix(panel)
        y_all = panel["residual"]

        # Train only on rows where the baseline and the true target are defined.
        mask = y_all.notna() & panel["y_hat_baseline"].notna()
        for entity, idx in panel.loc[mask].groupby("community_code").groups.items():
            X_e, y_e = X_all.loc[idx], y_all.loc[idx]
            # Not enough rows to fit a tree safely -> skip, fall back to 0 correction.
            if len(X_e) < 500:
                continue
            model = lgb.LGBMRegressor(**self.params)
            model.fit(X_e, y_e)
            self.models[entity] = model
        return self

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        X = self._build_feature_matrix(panel)
        out = pd.Series(0.0, index=panel.index, dtype=float)
        for entity, idx in panel.groupby("community_code").groups.items():
            model = self.models.get(entity)
            if model is None:
                continue
            out.loc[idx] = model.predict(X.loc[idx])
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrating model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EnergyConsumptionModel:
    """
    End-to-end day-ahead portfolio forecaster.

    Usage (pandas, for the leakage-free CV in `time_series_cv()`)
        m = EnergyConsumptionModel()
        m.fit(train_consumption, train_weather_forecast, demand_forecast)
        y_hat = m.predict_pandas(valid_panel_with_features)

    Databricks usage (scoring job)
        m = EnergyConsumptionModel()
        m.predict(spark_df, predict_start, predict_end)   # see final cell
    """
    k_big_clients: int = 30
    extractor: BigClientExtractor = field(default=None)
    residual: ResidualLGBM = field(default=None)

    # ------- pandas fit / predict (used by the CV) -----------------------
    def fit(
        self,
        consumption_df: pd.DataFrame,
        weather_forecast_df: pd.DataFrame,
        demand_forecast_df: pd.DataFrame,
    ) -> "EnergyConsumptionModel":
        # 1) Extract big clients from training window only.
        self.extractor = BigClientExtractor(k=self.k_big_clients).fit(consumption_df)
        df = self.extractor.transform(consumption_df)

        # 2) Build per-entity panel + baseline + weather + calendar.
        panel = build_entity_panel(df, demand_forecast_df)
        panel = attach_baseline(panel)
        panel = _attach_weather_forecast(panel, weather_forecast_df)
        panel = _add_calendar_features(panel)

        # 3) Fit residual LightGBM per entity.
        self.residual = ResidualLGBM().fit(panel)
        return self

    def predict_pandas(self, panel: pd.DataFrame) -> pd.Series:
        """
        `panel` must carry at least: community_code, datetime_local, alpha_D7,
        n_clients_D2, D_mw, plus the weather forecast & calendar columns.
        """
        eps_hat = self.residual.predict(panel)
        y_hat   = panel["y_hat_baseline"].fillna(0.0) + eps_hat.fillna(0.0)
        return y_hat.rename("prediction")

    # ------- Spark-facing contract (Databricks scoring job) ---------------
    def predict(self, df, predict_start, predict_end):
        """
        Databricks scoring entry point.

        Args
        ----
        df             : PySpark DataFrame (datathon.shared.client_consumption)
        predict_start  : str, inclusive
        predict_end    : str, exclusive

        Returns
        -------
        PySpark DataFrame with columns (datetime_15min, prediction).
        """
        # We convert the minimal slice we need to pandas; the raw consumption
        # table is far too large to `.toPandas()` blindly.
        from pyspark.sql import functions as F

        spark = df.sparkSession
        demand_sdf   = spark.table("datathon.shared.demand_forecast")
        weather_sdf  = spark.table("datathon.belmonte_hunters.openmeteo_previous_day_2_runs_15_min")

        # Aggregate per (community, 15-min) inside Spark *before* collecting.
        agg_sdf = (
            df.groupBy("community_code", "datetime_local")
              .agg(F.sum("active_kw").alias("total_kw"),
                   F.countDistinct("client_id").alias("n_clients"))
        )
        panel_pd   = agg_sdf.toPandas()
        demand_pd  = (demand_sdf.select(F.col("datetime_utc").alias("datetime_local"),
                                        F.col("value").alias("D_mw"))
                                .toPandas())
        weather_pd = weather_sdf.toPandas()
        cons_pd    = df.select("client_id", "community_code", "datetime_local", "active_kw").toPandas()

        # Fit on everything strictly before predict_start.
        train_mask = pd.to_datetime(cons_pd["datetime_local"]) < pd.Timestamp(predict_start)
        self.fit(
            cons_pd.loc[train_mask],
            weather_pd,
            demand_pd,
        )

        # Build the evaluation panel (we already have aggregates).
        eval_panel = panel_pd.merge(demand_pd, on="datetime_local", how="left")
        eval_panel = self.extractor.transform(eval_panel) if "client_id" in eval_panel.columns else eval_panel
        eval_panel = attach_baseline(eval_panel)
        eval_panel = _attach_weather_forecast(eval_panel, weather_pd)
        eval_panel = _add_calendar_features(eval_panel)

        horizon = (
            (pd.to_datetime(eval_panel["datetime_local"]) >= pd.Timestamp(predict_start)) &
            (pd.to_datetime(eval_panel["datetime_local"]) <  pd.Timestamp(predict_end))
        )
        eval_panel = eval_panel.loc[horizon].copy()
        eval_panel["y_hat"] = self.predict_pandas(eval_panel)

        total = (
            eval_panel.groupby("datetime_local", as_index=False)["y_hat"]
                      .sum()
                      .rename(columns={"datetime_local": "datetime_15min",
                                       "y_hat":          "prediction"})
        )
        return spark.createDataFrame(total).select("datetime_15min", "prediction")


# ═══════════════════════════════════════════════════════════════════════════
# Leakage-free k-fold cross-validation (TimeSeriesSplit on the target date)
# ═══════════════════════════════════════════════════════════════════════════

def time_series_cv(
    consumption_df: pd.DataFrame,
    weather_forecast_df: pd.DataFrame,
    demand_forecast_df: pd.DataFrame,
    n_splits: int = 5,
    gap_days: int = 2,
    k_big_clients: int = 30,
) -> pd.DataFrame:
    """
    Walk-forward validation at the 15-min portfolio level.

    `gap_days` is the safety buffer between the last training timestamp and the
    first validation timestamp. We require >= 2 because the baseline uses a
    D-2 snapshot of the client count -- bridging that gap prevents features
    built at the training boundary from relying on the validation window.

    Returns a DataFrame with per-fold MAE (portfolio kW) for auditing.
    """
    from sklearn.model_selection import TimeSeriesSplit

    df = consumption_df.copy()
    df["datetime_local"] = pd.to_datetime(df["datetime_local"])
    days = np.sort(df["datetime_local"].dt.normalize().unique())

    tss = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_days_idx, valid_days_idx) in enumerate(tss.split(days)):
        train_days = days[train_days_idx]
        valid_days = days[valid_days_idx]
        # enforce the gap
        valid_days = valid_days[valid_days >= train_days.max() + np.timedelta64(gap_days, "D")]
        if len(valid_days) == 0:
            continue

        train_mask = df["datetime_local"].dt.normalize().isin(train_days)
        valid_mask = df["datetime_local"].dt.normalize().isin(valid_days)

        model = EnergyConsumptionModel(k_big_clients=k_big_clients).fit(
            df.loc[train_mask],
            weather_forecast_df,
            demand_forecast_df,
        )

        # Rebuild the validation panel the same way .predict would
        df_extracted = model.extractor.transform(df.loc[train_mask | valid_mask])
        panel = build_entity_panel(df_extracted, demand_forecast_df)
        panel = attach_baseline(panel)
        panel = _attach_weather_forecast(panel, weather_forecast_df)
        panel = _add_calendar_features(panel)
        panel = panel.loc[panel["datetime_local"].dt.normalize().isin(valid_days)]

        y_hat_total = (
            panel.assign(y_hat=model.predict_pandas(panel))
                 .groupby("datetime_local", as_index=False)["y_hat"].sum()
        )
        y_true_total = (
            df.loc[valid_mask]
              .groupby("datetime_local", as_index=False)["active_kw"].sum()
              .rename(columns={"active_kw": "y_true"})
        )
        merged = y_hat_total.merge(y_true_total, on="datetime_local", how="inner")
        mae = (merged["y_true"] - merged["y_hat"]).abs().mean()

        results.append({
            "fold":         fold,
            "train_start":  train_days.min(),
            "train_end":    train_days.max(),
            "valid_start":  valid_days.min(),
            "valid_end":    valid_days.max(),
            "mae_kw":       mae,
            "n_points":     len(merged),
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# Databricks scoring cell (unchanged contract)
# ═══════════════════════════════════════════════════════════════════════════
# The block below is a 1:1 copy of the final scoring cell from the original
# submission notebook. It is what the Axpo / Databricks scoring job expects
# to run. Nothing above this line should be edited for a resubmission --
# only the EnergyConsumptionModel implementation changes how predictions are
# produced.

if __name__ == "__main__" and "spark" in globals():   # Databricks entry point
    # >>> Set your team name <<<
    TEAM_NAME = "belmonte_hunters"

    # ============================================================
    # DO NOT CHANGE THIS CELL - submission will break
    # ============================================================
    SCORING_JOB_ID = 155971627612104

    dbutils.widgets.text("mode", "interactive")          # noqa: F821
    _MODE = dbutils.widgets.get("mode").strip()           # noqa: F821

    if _MODE == "score":
        from pyspark.sql import functions as _F

        _predict_start = dbutils.widgets.get("predict_start").strip()   # noqa: F821
        _predict_end   = dbutils.widgets.get("predict_end").strip()     # noqa: F821

        _full_df     = spark.table("datathon.shared.client_consumption")  # noqa: F821
        _model       = EnergyConsumptionModel()
        _predictions = _model.predict(_full_df, _predict_start, _predict_end)

        _predictions_table = "datathon.evaluation.submissions"
        (
            _predictions
            .withColumn("team_name",    _F.lit(TEAM_NAME))
            .withColumn("submitted_at", _F.current_timestamp())
            .select("team_name", "datetime_15min", "prediction", "submitted_at")
            .write.mode("overwrite").saveAsTable(_predictions_table)
        )
        print(f"Wrote {_predictions.count():,} predictions to {_predictions_table}")
        dbutils.notebook.exit("ok")   # noqa: F821
