# WUR Streamlit App

# CS

Interaktivní **Streamlit** aplikace pro prohlížení **historie a predikcí** žebříčku World University Rankings (WUR), srovnání více institucí a validaci modelů (**MAE/RMSE**) napříč horizonty. Aplikace pracuje nad vlastním datasetem a nad artefakty z validací/backtestu.

---

## Funkce

- **Predikce (01_Predikce)**  
  - výběr instituce, **slider „Začátek období“** a **max. horizont K**  
  - graf *Historie & Predikce skóre* + **export PNG**  
  - graf *Historie & Predikce pořadí* + **export PNG**  
  - tabulka budoucích `y_hat` a **agregované metriky** pro danou instituci/model

- **Komparace (02_Komparace)**  
  - výběr více institucí  
  - overlay historie a predikcí (automaticky volí **vítězný model per škola**)  
  - **MAE/RMSE podle horizontu K** + export CSV

- **Validace (03_Validace)**  
  - **MAE/RMSE** po *model × K* ve třech agregacích: **makro-průměr**, **medián**, **vážený průměr**  
  - diagnostika s počty párů **N** v jednotlivých buňkách  
  - přehled **vítězných modelů** dle pravidel (winner flag / min MAE / min RMSE)

- **UI doplňky**  
  - společný **sidebar** s logem (viz `src/wur/ui.py`)  
  - download buttony pro grafy a tabulky  
  - robustní práce s chybějícími daty (NaN, rank-bandy apod.)

---

## Požadavky

- Python **3.10+** (doporučeno 3.11)  
- `pip install -r requirements.txt`

---

## Data & proměnné prostředí

| Proměnná            | Význam                                                                 | Výchozí |
|---------------------|------------------------------------------------------------------------|---------|
| `RAW_DIR`           | Kořen pro WUR dataset (`wur_dataset.(parquet|csv)`)                    | `data/clean` |
| `DATA_DIR`          | Kořen pro artefakty (`forecast.*`, `overall.*`, …)                     | `.` |
| `FORECAST_PATH`     | Přesná cesta k forecast souboru (přebije hledání v `DATA_DIR`)         | – |
| `BACKTEST_RAW_PATH` | Cesta k raw backtestu (MAE na úrovni párů), pro Validaci               | `outputs/backtest/backtest_raw_results.csv` |
| `APP_ENV`           | Režim aplikace (`dev`/`prod`) – jen pro logování                       | `prod` |

- **WUR dataset** se očekává v `RAW_DIR/wur_dataset.parquet` (nebo `.csv`).
- **Forecast/overall** se hledají v `DATA_DIR` (případně přes `FORECAST_PATH`).
- **Backtest** (CSV/Parquet) umožní zobrazit mediány a vážené průměry; bez něj se použije fallback (makro-průměry).

---

## Spuštění lokálně

```bash
# 1) instalace
pip install -r requirements.txt

# 2) (doporučeno) nastavení cest
export RAW_DIR="data/clean"
export DATA_DIR="outputs"
export BACKTEST_RAW_PATH="outputs/backtest/backtest_raw_results.csv"

# 3) spuštění
streamlit run app.py

Aplikace poběží na http://localhost:8501.


## Metriky

* MAE – průměrná absolutní chyba (v bodech skóre).

* RMSE – odmocnina z průměru kvadrátů chyb (citlivější na outliery).

Aplikace metriky:

* použije z forecast, pokud jsou k dispozici,

* jinak je dopočítá spárováním s historií (origin_year + K = target_year),

* nebo využije raw backtest, je-li k dispozici (zejména pro Validaci).


---

# ENG

An interactive **Streamlit** app to explore **history and forecasts** of World University Rankings (WUR), compare multiple institutions, and validate models (**MAE/RMSE**) across forecast horizons. The app works with your dataset and training/backtesting artifacts.

---

## Features

- **Prediction (01_Predikce)**  
  - institution selector, **“Start year”** slider, and **max horizon K**  
  - *History & Forecast of score* plot + **PNG export**  
  - *History & Forecast of rank* plot + **PNG export**  
  - table of future `y_hat` and **aggregated metrics** for the chosen institution/model

- **Comparison (02_Komparace)**  
  - select multiple institutions  
  - overlay of history and forecasts (automatically picks the **winning model per institution**)  
  - **MAE/RMSE by horizon K** + CSV export

- **Validation (03_Validace)**  
  - **MAE/RMSE** for *model × K* in three aggregations: **macro average**, **median**, **weighted average**  
  - diagnostics with the number of pairs **N** per cell  
  - **winning models** overview (winner flag / min MAE / min RMSE)

- **UI extras**  
  - shared **sidebar** with a logo (`src/wur/ui.py`)  
  - download buttons for plots and tables  
  - robust handling of missing values (NaN, rank-bands, etc.)

---

## Requirements

- Python **3.10+** (3.11 recommended)  
- `pip install -r requirements.txt`

---

## Data & environment variables

| Variable             | Meaning                                                                 | Default |
|----------------------|-------------------------------------------------------------------------|---------|
| `RAW_DIR`            | Root for WUR dataset (`wur_dataset.(parquet|csv)`)                      | `data/clean` |
| `DATA_DIR`           | Root for artifacts (`forecast.*`, `overall.*`, …)                       | `.` |
| `FORECAST_PATH`      | Exact forecast file path (overrides lookup in `DATA_DIR`)               | – |
| `BACKTEST_RAW_PATH`  | Raw backtest path (pair-level MAE), used by Validation                  | `outputs/backtest/backtest_raw_results.csv` |
| `APP_ENV`            | App mode (`dev`/`prod`) – logging only                                  | `prod` |

- **WUR dataset** expected at `RAW_DIR/wur_dataset.parquet` (or `.csv`).  
- **Forecast/overall** are searched under `DATA_DIR` (or set `FORECAST_PATH`).  
- **Backtest** (CSV/Parquet) enables medians and weighted averages; otherwise a fallback macro average is used.

---

## Run locally

```bash
# 1) install
pip install -r requirements.txt

# 2) (recommended) set paths
export RAW_DIR="data/clean"
export DATA_DIR="outputs"
export BACKTEST_RAW_PATH="outputs/backtest/backtest_raw_results.csv"

# 3) start
streamlit run app.py
The app runs at http://localhost:8501.

## Metrics

* MAE – mean absolute error (same units as the score).

* RMSE – root mean squared error (more sensitive to outliers).

The app:

* uses metrics from forecast if present,

* otherwise recomputes them by pairing with history (origin_year + K = target_year),

* or falls back to raw backtest where available (esp. on Validation page).