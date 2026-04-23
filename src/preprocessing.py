"""
preprocessing1.py — day-ahead training matrix with target = alpha.
"""
from pyspark.sql import functions as F, Window
import math


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — panel (unchanged)
# ═══════════════════════════════════════════════════════════════════════════
def build_community_timestep_panel(df):
    first_seen = (df.groupBy("client_id")
                    .agg(F.min("datetime_utc").alias("_first_dt")))
    df = (df.join(first_seen, "client_id", "left")
            .withColumn("_is_new",
                        (F.col("datetime_utc") == F.col("_first_dt")).cast("int"))
            .withColumn("_is_missing", F.col("active_kw").isNull().cast("int")))
    return (
        df.groupBy("community_code", "datetime_utc")
          .agg(F.first("datetime_local").alias("datetime_local"),
               F.countDistinct("client_id").alias("n_clients"),
               F.sum("_is_missing").alias("n_missing"),
               F.sum("_is_new").alias("n_new_clients"),
               F.sum("active_kw").alias("total_kw"),
               F.mean("active_kw").alias("mean_kw"),
               F.expr("percentile_approx(active_kw, 0.5)").alias("median_kw"),
               F.stddev("active_kw").alias("std_kw"),
               F.min("active_kw").alias("min_kw"),
               F.max("active_kw").alias("max_kw"))
          .withColumn("n_reporting", F.col("n_clients") - F.col("n_missing"))
          .withColumn("avg_kw_per_client",
                      F.when(F.col("n_clients")  > 0, F.col("total_kw") / F.col("n_clients")))
          .withColumn("avg_kw_reporting",
                      F.when(F.col("n_reporting") > 0, F.col("total_kw") / F.col("n_reporting")))
    )


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — alpha dataset
# ═══════════════════════════════════════════════════════════════════════════
def build_alpha_dataset(
    raw_df,
    demand_forecast,
    weather,
    holidays,
    cutoff_hour_local: int = 12,
    min_history_days: int = 28,
    lag_days=(2, 7, 14, 21, 28),
):
    assert all(d >= 2 for d in lag_days), "All lag_days must be ≥ 2 (cutoff safety)."

    # ── 1. Panel ─────────────────────────────────────────────────────────
    panel = build_community_timestep_panel(raw_df)

    # ── 2. Demand forecast join (hourly UTC) ─────────────────────────────
    D = (demand_forecast.select(
            F.date_trunc("hour", F.col("datetime_utc")).alias("_h"),
            F.col("value").alias("D_mw")))
    panel = (panel.withColumn("_h", F.date_trunc("hour", F.col("datetime_utc")))
                  .join(D, "_h", "left")
                  .drop("_h"))

    # ── 2b. Weather join (LOCAL time, 15-min) ────────────────────────────
    w = (weather
         .drop("province")
         .withColumnRenamed("time", "datetime_local")
         .dropDuplicates(["community_code", "datetime_local"]))
    panel = panel.join(w, ["community_code", "datetime_local"], "left")

    # ── 3. Alpha ─────────────────────────────────────────────────────────
    panel = panel.withColumn(
        "alpha",
        F.when((F.col("n_clients") > 0) &
               (F.col("D_mw") > 0) &
               F.col("total_kw").isNotNull(),
               F.col("total_kw") / (F.col("n_clients") * F.col("D_mw"))))

    # ── 4. target_date ──────────────────────────────────────────────────
    p = panel.withColumn("target_date", F.to_date("datetime_local"))

    # ── 4b. Holiday flag ────────────────────────────────────────────────
    hol_dates = holidays.select("date").distinct()
    p = (p.join(hol_dates,
                p["target_date"] == hol_dates["date"], "left")
         .withColumn("is_holiday", F.when(F.col("date").isNotNull(), 1).otherwise(0))
         .drop("date"))

    # ── 5. Lags ─────────────────────────────────────────────────────────
    X = p
    lag_cols = []
    for N in lag_days:
        name = f"alpha_lag_{N}d"
        lag_cols.append(name)
        lagged = (p.select(
            "community_code",
            (F.col("datetime_utc") + F.expr(f"INTERVAL {N} DAYS")).alias("datetime_utc"),
            F.col("alpha").alias(name)))
        X = X.join(lagged, ["community_code", "datetime_utc"], "left")

    s = sum(F.coalesce(F.col(c), F.lit(0.0)) for c in lag_cols)
    n = sum(F.when(F.col(c).isNotNull(), 1).otherwise(0) for c in lag_cols)
    X = X.withColumn("alpha_lag_mean", F.when(n > 0, s / n))

    # ── 6. Snapshot at D-1 11:45 local ──────────────────────────────────
    SNAP_H, SNAP_M = cutoff_hour_local - 1, 45
    order_utc = F.col("datetime_utc").cast("long")
    w_24h = Window.partitionBy("community_code").orderBy(order_utc)\
        .rangeBetween(-24 * 3600 + 1, 0)
    w_7d  = Window.partitionBy("community_code").orderBy(order_utc)\
        .rangeBetween(-7 * 24 * 3600 + 1, 0)

    snap = (p.withColumn("sn_alpha_last",     F.col("alpha"))
             .withColumn("sn_alpha_mean_24h", F.avg("alpha").over(w_24h))
             .withColumn("sn_alpha_mean_7d",  F.avg("alpha").over(w_7d))
             .withColumn("sn_alpha_std_7d",   F.stddev("alpha").over(w_7d))
             .withColumn("n_clients_at_cutoff", F.col("n_clients"))
             .filter((F.hour("datetime_local")   == SNAP_H) &
                     (F.minute("datetime_local") == SNAP_M))
             .withColumn("target_date", F.date_add(F.to_date("datetime_local"), 1))
             .select("community_code", "target_date",
                     "sn_alpha_last", "sn_alpha_mean_24h",
                     "sn_alpha_mean_7d", "sn_alpha_std_7d",
                     "n_clients_at_cutoff"))

    X = X.join(snap, ["community_code", "target_date"], "left")

    # ── 7. Calendar ─────────────────────────────────────────────────────
    TAU = 2 * math.pi
    X = (X.withColumn("hour",       F.hour("datetime_local"))
          .withColumn("qhour",      (F.minute("datetime_local") / 15).cast("int"))
          .withColumn("mod",        F.col("hour") * 4 + F.col("qhour"))
          .withColumn("dow",        F.dayofweek("datetime_local"))
          .withColumn("month",      F.month("datetime_local"))
          .withColumn("doy",        F.dayofyear("datetime_local"))
          .withColumn("is_weekend", F.col("dow").isin(1, 7).cast("int"))
          .withColumn("sin_mod",    F.sin(F.lit(TAU) * F.col("mod") / 96))
          .withColumn("cos_mod",    F.cos(F.lit(TAU) * F.col("mod") / 96))
          .withColumn("sin_dow",    F.sin(F.lit(TAU) * F.col("dow") / 7))
          .withColumn("cos_dow",    F.cos(F.lit(TAU) * F.col("dow") / 7)))

    # ── 8. Label ────────────────────────────────────────────────────────
    X = X.withColumn("y", F.col("alpha"))

    # ── 9. History filter ───────────────────────────────────────────────
    first_date = (panel.groupBy("community_code")
                       .agg(F.min(F.to_date("datetime_local")).alias("_first")))
    X = (X.join(first_date, "community_code", "left")
          .filter(F.datediff(F.col("target_date"), F.col("_first")) >= min_history_days)
          .filter(F.col("y").isNotNull())
          .drop("_first"))

    # ── 10. Project ─────────────────────────────────────────────────────
    bookkeeping = ["community_code", "target_date", "datetime_utc",
                   "datetime_local", "total_kw", "n_clients"]
    features = list(_BASE_FEATURES) + (list(_WEATHER_FEATURES) if weather is not None else [])
    return X.select(*bookkeeping, "y", *features)


# ═══════════════════════════════════════════════════════════════════════════
# Feature catalog (module-level so notebooks can import)
# ═══════════════════════════════════════════════════════════════════════════
_BASE_FEATURES = [
    "alpha_lag_2d", "alpha_lag_7d", "alpha_lag_14d", "alpha_lag_21d", "alpha_lag_28d",
    "alpha_lag_mean",
    "sn_alpha_last", "sn_alpha_mean_24h", "sn_alpha_mean_7d", "sn_alpha_std_7d",
    "n_clients_at_cutoff",
    "D_mw",
    "hour", "qhour", "mod", "dow", "month", "doy", "is_weekend",
    "sin_mod", "cos_mod", "sin_dow", "cos_dow",
    "is_holiday",
]

_WEATHER_FEATURES = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "precipitation", "cloud_cover",
    "wind_speed_10m", "shortwave_radiation", "diffuse_radiation", "is_day",
]

FEATURE_COLS     = _BASE_FEATURES + _WEATHER_FEATURES
CATEGORICAL_COLS = ["community_code", "hour", "qhour", "dow", "month", "is_weekend", "is_day", "is_holiday"]
