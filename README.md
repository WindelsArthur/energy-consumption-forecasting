# Iberia Retail Energy Consumption Forecasting
### ETH Datathon 2025 — Axpo × Databricks challenge · 🥈 2nd place

This repository documents the day-ahead electricity consumption forecasting
pipeline we built at the **ETH Datathon 2025**, the largest data-science /
machine-learning competition in Switzerland (hosted at ETH Zürich). The
challenge — titled *"Iberia Retail Consumption Forecasting"* — was designed
by **Axpo** in partnership with **Databricks**, and our solution was ranked
**2nd overall** among the participating teams.

> **No data in this repository.** The competition dataset (client-level
> metered consumption, Spain demand / PV / wind forecasts, weather, etc.) is
> covered by an Axpo non-disclosure agreement and lived exclusively inside a
> Databricks Unity Catalog. What you will find here is the **idea, the
> reasoning, and the pipeline** that produced our best score — presented in a
> way that is faithful to the submitted notebook but strictly leakage-free.

---

## 👥 Team Belmonte Hunters

|   | Member | LinkedIn |
|---|--------|----------|
| 👤 | **Arthur Vianna**    | <https://www.linkedin.com/in/arthur-vianna/> |
| 👤 | **Yanis Fallet**     | <https://www.linkedin.com/in/yanis-fallet/> |
| 👤 | **Nicolaj Thomsen**  | <https://www.linkedin.com/in/nicolaj-thomsen/> |
| 👤 | **Arthur Windels**   | <https://www.linkedin.com/in/arthur-windels/> |

*(LinkedIn handles are placeholders — each team member can replace with their
own profile URL.)*

---

## 🏁 The Challenge

> "You take on the role of a retail energy supplier operating in the Iberian
> market. Your objective is to develop a forecasting model that minimises the
> consumption forecast error of your client portfolio in order to reduce
> imbalance." — *Axpo challenge brief*

### Context

In liberalised electricity markets (Iberia operates under the **MIBEL** day-
ahead auction), retailers buy electricity on wholesale markets and resell it
to end consumers. They must declare their expected consumption **before 12:00
on day D-1** for every 15-minute interval of day D. Any deviation between
declared and realised consumption is called an **imbalance**, and is settled
at prices that are always *less* favourable than the day-ahead price:

| Situation | Outcome |
|-----------|---------|
| Procured too much → *long* | sell the surplus back at a discount |
| Procured too little → *short* | buy the deficit at a premium |

The larger the forecast error, the larger the imbalance cost. **Minimising
MAE is therefore the direct lever for P&L.**

### Task

- Produce day-ahead forecasts (before 12:00 of D-1) of the **total portfolio
  active power** at 15-minute granularity, for every 15-min interval of day D.
- Metric: **Mean Absolute Error (MAE)** over the test window
  *1 Dec 2025 – 28 Feb 2026*.
- Training data: client-level consumption for **1 Jan 2025 – 30 Nov 2025**.
- Free to enrich with any public dataset; no data leakage allowed.

### Data

All tables were provided in Databricks Unity Catalog (`datathon.*`):

| Table | Description |
|-------|-------------|
| `datathon.shared.client_consumption`      | Per-client `active_kw` at 15 min, 2025-01 → 2025-11 |
| `datathon.shared.demand_forecast`         | Spain day-ahead demand forecast (REE) |
| `datathon.shared.pv_production_forecast`  | PV production forecast |
| `datathon.shared.wind_production_forecast`| Wind production forecast |

---

## 💡 Our Solution in One Paragraph

Modelling raw `active_kw` directly is hard: the portfolio roughly **doubled**
over 2025 (client count grew from ≈4.4k to ≈8.4k), consumption distributions
differ by more than an order of magnitude between regions, and a handful of
large industrial clients dominate a few communities. Rather than fighting all
of this with a monolithic regressor, we **re-target the problem** to a
quantity that is almost stationary:

$$
\alpha(c, t) \;=\; \frac{\sum_{j\,\in\,c}\, y_j(t)}{n(c,t)\;\cdot\;\hat D_{\text{Spain}}(t)}
$$

the per-client share of Spain's day-ahead demand forecast, per community.
The denominator absorbs seasonality, calendar effects, and the dominant
weather signal *for free* (REE's forecast already does that); the division by
`n(c,t)` neutralises portfolio growth.

Our **baseline** — the model that actually scored on the leaderboard — is a
structural estimator with **zero fitted parameters**:

$$
\hat y_{\text{portfolio}}(t)
\;=\;
\sum_{c}\; \alpha(c,\,D-7)\;\cdot\;n(c,\,D-2)\;\cdot\;\hat D_{\text{Spain}}(t)
$$

which simply extrapolates last week's share forward. We tried extensively to
beat it with learned residual models (LightGBM per community, with weather
forecasts and calendar encodings; big-client extraction; clustering; stacking)
but **no variant produced a gain large enough to justify the added
complexity**, so we submitted the baseline.

> See `slides/presentation.pdf` for the full narrative.

---

## 📊 Key plots

### Raw consumption is heterogeneous across the 17 communities
![Consumption histograms](assets/consumption_histograms.png)

Daily totals span more than an order of magnitude between regions — any model
on raw `active_kw` is dominated by the largest ones.

### Portfolio roughly doubled over the training window
![Portfolio growth](assets/portfolio_growth.png)

A model on total consumption would absorb this trend and extrapolate it
blindly. Normalising by `n(c, t)` solves that.

### α is near-stationary for most communities
![Alpha per community](assets/alpha_per_community.png)

Monthly-averaged per-client share of Spain demand. Most communities sit in a
narrow band → α is much easier to predict than `active_kw`.

### Final forecast vs. observed (held-out slice of training)
![Final prediction](assets/final_prediction.png)

---

## 🗂️ Repository layout

```
energy_consumption_forecasting/
├── README.md                           ← you are here
├── src/
│   ├── preprocessing.py                ← Spark pipeline: panel + alpha + lags + weather/holiday joins
│   └── submission.py                   ← clean reference implementation (see below)
├── notebooks/                          ← original Databricks notebooks, renamed for clarity
│   ├── data_exploration_1.ipynb
│   ├── data_exploration_2.ipynb
│   ├── exploration_preprocessing_2.ipynb
│   ├── model_exploration_1.ipynb
│   ├── model_exploration_2.ipynb
│   ├── model_exploration_3.ipynb
│   └── model_exploration_4.ipynb
└── assets/                             ← plots used in the README
```

The notebooks are shipped **as-is** from the competition workspace. Because
the Databricks data is not available outside the competition, the notebooks
are read-only artefacts that document what we explored, not runnable code.

### `src/submission.py`

A clean, self-contained rewrite of the submitted `EnergyConsumptionModel`
class that:

1. **Identifies big clients** from the training window only (no leakage into
   validation or test).
2. **Computes α** per `(community, 15-min)` against the Spain demand forecast.
3. **Produces the baseline** `ŷ(t) = α(c, D-7) · n(c, D-2) · D̂_Spain(t)`.
4. **Fits one LightGBM per entity** on the residual `ε = y − ŷ_baseline`,
   using only **D-2 weather forecasts** and calendar encodings.
5. **Runs a leakage-free `TimeSeriesSplit` cross-validation** (`time_series_cv`)
   with a configurable gap between train and validation, large enough to
   avoid the D-2 / D-7 feature window overlapping the held-out fold.

The Databricks-facing `predict(df, predict_start, predict_end)` contract and
the final scoring cell are preserved verbatim, so the module remains
drop-in compatible with the competition scoring job.

### `src/preprocessing.py`

The original PySpark feature-engineering pipeline used in the notebooks:
builds the `(community × 15-min)` panel, joins the demand/weather/holiday
tables, computes α and its lags, and produces the D-1 11:45 snapshots used
by the learned models. Depends on PySpark and therefore only runs inside a
Databricks workspace.

---

## 🧪 Reproducing the pipeline (outside Databricks)

Because the source data is under NDA, there is no way to actually reproduce
the numbers outside the competition workspace. The code in `src/` is
structured so that, *given* a pandas dataframe of client consumption and
matching demand / weather forecasts, the pipeline runs end-to-end:

```python
from src.submission import EnergyConsumptionModel, time_series_cv

# consumption_df      : columns [client_id, community_code, datetime_local, active_kw]
# demand_forecast_df  : columns [datetime_local, D_mw]
# weather_forecast_df : columns [community_code, datetime_local, <weather columns>]

# Honest walk-forward MAE per fold
cv = time_series_cv(
    consumption_df, weather_forecast_df, demand_forecast_df,
    n_splits=5, gap_days=2,
)
print(cv)

# Fit on everything for the actual submission
model = EnergyConsumptionModel().fit(
    consumption_df, weather_forecast_df, demand_forecast_df,
)
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `lightgbm` (and `pyspark`
only for the Databricks entry point).

---

## 🙏 Acknowledgements

- **Axpo** for designing the challenge and sharing the portfolio data.
- **Databricks** for providing the workspace and compute.
- **Analytics Club at ETH** / **ETH Datathon organisers** for running the
  biggest data-science event in Switzerland.
- **REE** for the open Spain demand forecast, and **Open-Meteo** for the
  weather forecast API that powered our D-2 weather features.
