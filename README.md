# Vehicle Price Prediction — MLOps Pipeline on Azure ML

An end-to-end MLOps project that predicts the price of used cars with a Random Forest regressor. Every push to `main` triggers a GitHub Actions workflow that registers the data and environment in Azure Machine Learning, then runs a pipeline that prepares the data, tunes the model's hyperparameters, and registers the best model.


## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository structure](#repository-structure)
- [Dataset](#dataset)
- [Pipeline steps](#pipeline-steps)
- [CI/CD with GitHub Actions](#cicd-with-github-actions)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [Viewing results](#viewing-results)
- [Next steps](#next-steps)

---

## Overview

| Item | Details |
|---|---|
| **Goal** | Predict the selling price of a used car from its characteristics |
| **Problem type** | Regression |
| **Model** | `RandomForestRegressor` (scikit-learn) |
| **Tuning** | Azure ML sweep job (random search over `n_estimators` and `max_depth`) |
| **Metric** | Mean Squared Error (MSE) on the test set (lower is better) |
| **Tracking** | MLflow (parameters, metrics, and model artifacts) |
| **Platform** | Azure Machine Learning (CLI v2) |
| **Automation** | GitHub Actions |

---

## Architecture

```
 Developer pushes to main
          │
          ▼
 ┌─────────────────────────────┐
 │       GitHub Actions        │
 │  1. Log in to Azure         │
 │  2. Install Azure ML CLI    │
 │  3. Ensure compute exists   │
 │  4. Register dataset        │
 │  5. Register environment    │
 │  6. Submit pipeline job     │
 └──────────────┬──────────────┘
                │
                ▼
 ┌──────────────────────────────────────────────────────────┐
 │              Azure ML pipeline (newpipeline.yml)         │
 │                                                          │
 │  used_cars.csv ──► prep_data ──► sweep_step ──► register_model
 │                   (clean,       (16+ Random    (best model
 │                    encode,       Forest trials, registered as
 │                    80/20 split)  min. MSE)     used_cars_price_
 │                                                prediction_model)
 └──────────────────────────────────────────────────────────┘
```

---

## Repository structure

```
Vehicle-price-prediction/
├── .github/
│   └── workflows/
│       └── train-AzureML-pipeline.yml   # CI/CD: runs the Azure ML pipeline on push
├── data/
│   └── used_cars.csv                    # Raw dataset (200 rows)
├── data-science/
│   ├── environment/
│   │   └── train-conda.yml              # Python/Conda dependencies for training
│   └── src/
│       ├── prep.py                      # Data cleaning, encoding, train/test split
│       ├── train.py                     # Random Forest training + MSE evaluation
│       └── register.py                  # Registers the best model in Azure ML
├── mlops/
│   └── azureml/
│       └── train/
│           ├── data.yml                 # Data asset definition (used-cars-data)
│           ├── train-env.yml            # Environment definition (used-cars-train-env)
│           ├── train.yml                # Command component used as the sweep trial
│           └── newpipeline.yml          # Pipeline: prep → sweep → register
├── config-infra-prod.yml                # Infrastructure settings (region, names)
├── .gitignore
└── README.md
```

---

## Dataset

`data/used_cars.csv` contains **200 used cars** with 6 features and 1 target.

| Column | Type | Description |
|---|---|---|
| `Segment` | Categorical | `luxury segment` (86 cars) or `non-luxury segment` (114 cars) |
| `Kilometers_Driven` | Numeric | Total distance driven (km) |
| `Mileage` | Numeric | Fuel efficiency (km per litre) |
| `Engine` | Numeric | Engine displacement (cc) |
| `Power` | Numeric | Maximum power (bhp) |
| `Seats` | Numeric | Number of seats |
| `price` | Numeric | **Target**: selling price of the car |

Summary of the numeric columns:

| Column | Mean | Median | Max |
|---|---|---|---|
| Kilometers_Driven | 56,216 | 54,000 | 262,000 |
| Mileage | 17.3 | 17.1 | 28.4 |
| Engine | 1,830 | 1,875 | 5,461 |
| Power | 139.5 | 126.3 | 500 |
| Seats | 5.2 | 5 | 8 |
| price | 34.9 | 19.1 | 136.0 |

---

## Pipeline steps

The pipeline is defined in [`mlops/azureml/train/newpipeline.yml`](mlops/azureml/train/newpipeline.yml) and has three steps.

### 1. `prep_data` — data preparation ([`prep.py`](data-science/src/prep.py))

- Reads the raw CSV registered as the `used-cars-data` data asset.
- Cleans the `Segment` column (trims spaces, lower-cases) so that spelling variations don't create extra categories.
- Label-encodes `Segment` into numbers.
- Splits the data into **80 % training / 20 % test** (`test_train_ratio: 0.2`, `random_state=42`).
- Writes `train.csv` and `test.csv` and logs the size of each split to MLflow.

### 2. `sweep_step` — hyperparameter tuning ([`train.py`](data-science/src/train.py), [`train.yml`](mlops/azureml/train/train.yml))

Each trial trains a `RandomForestRegressor`, predicts on the test set, and logs **MSE** to MLflow. Azure ML's sweep job tries combinations from this search space:

| Hyperparameter | Values |
|---|---|
| `n_estimators` (number of trees) | 10, 20, 30, 50 |
| `max_depth` (maximum tree depth) | 5, 10, 15, 20 |

| Sweep setting | Value |
|---|---|
| Sampling algorithm | Random |
| Objective | Minimize `MSE` |
| Max total trials | 20 |
| Max concurrent trials | 10 |
| Timeout | 2 hours |

The model from the best trial (lowest MSE) is passed to the next step.

### 3. `register_model` — model registration ([`register.py`](data-science/src/register.py))

- Loads the best model from the sweep.
- Registers it in the Azure ML model registry as **`used_cars_price_prediction_model`** (a new version on every run).
- Writes `model_info.json` containing the registered model name and version.

---

## CI/CD with GitHub Actions

The workflow [`.github/workflows/train-AzureML-pipeline.yml`](.github/workflows/train-AzureML-pipeline.yml) runs:

- automatically on every **push to `main`**, and
- manually from the **Actions** tab using **Run workflow**.

| Step | What it does |
|---|---|
| Checkout | Downloads the repository onto the GitHub runner |
| Log in to Azure | Authenticates with the service principal in the `AZURE_CREDENTIALS_AA` secret |
| Install Azure ML CLI | Adds the `ml` extension to the Azure CLI and sets the default resource group and workspace |
| Create compute cluster | Creates `cpu-cluster` (Standard_A2_v2, 0–2 nodes) if it does not exist yet |
| Register dataset | Uploads `used_cars.csv` as a new version of `used-cars-data` |
| Register environment | Builds a new version of `used-cars-train-env` from `train-conda.yml` |
| Run training pipeline | Submits `newpipeline.yml` and streams its logs until it finishes |

---

## Setup

### Prerequisites

- An Azure subscription with an **Azure Machine Learning workspace**
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) with the ML extension (`az extension add -n ml`)
- A GitHub account with access to this repository

### 1. Clone the repository

```bash
git clone https://github.com/aamele/Vehicle-price-prediction.git
cd Vehicle-price-prediction
```

### 2. Update the Azure settings

Set your own resource group and workspace names in:

- `.github/workflows/train-AzureML-pipeline.yml` → `RESOURCE_GROUP` and `WORKSPACE`
- `config-infra-prod.yml` → `resource_group` and `aml_workspace`

### 3. Create a service principal for GitHub

```bash
az login --use-device-code
SUB=$(az account show --query id -o tsv)
az ad sp create-for-rbac --name "github-mlops" --role contributor \
  --scopes /subscriptions/$SUB/resourceGroups/<your-resource-group> --json-auth
```

Copy the full JSON output. It is a secret: never commit it to the repository.

### 4. Add the secret to GitHub

In the repository, go to **Settings → Secrets and variables → Actions → New repository secret**:

- **Name:** `AZURE_CREDENTIALS_AA`
- **Value:** the JSON from the previous step

### 5. Choose the compute

The pipeline's default compute is set in `newpipeline.yml`:

```yaml
settings:
  default_compute: azure-aa       # compute instance (must be running)
  # default_compute: azureml:cpu-cluster  # compute cluster (starts automatically)
```

A compute **instance** must be started in Azure ML Studio before a run. A compute **cluster** starts automatically, scales to zero when idle, and can run trials in parallel.

---

## Running the pipeline

**Automatically:** push any change to `main`:

```bash
git add .
git commit -m "Describe your change"
git pull --rebase
git push
```

**Manually from GitHub:** open **Actions → Train model in Azure ML → Run workflow**.

**From your own machine** (after `az login`):

```bash
az configure --defaults group=<resource-group> workspace=<workspace>
az ml data create --file mlops/azureml/train/data.yml
az ml environment create --file mlops/azureml/train/train-env.yml
az ml job create --file mlops/azureml/train/newpipeline.yml --web
```

---

## Viewing results

| Where | What you'll find |
|---|---|
| GitHub → **Actions** | Status of each workflow run and its logs |
| [Azure ML Studio](https://ml.azure.com) → **Jobs** → `used-cars-price-pipeline` | The pipeline graph, the logs of each step, and the sweep trials |
| Sweep step → **Trials** | MSE for every hyperparameter combination, and the best trial |
| **Models** → `used_cars_price_prediction_model` | Every registered model version |

---

## Next steps

- Deploy the registered model to a **managed online endpoint** for real-time predictions.
- Add data validation and unit tests that run before training.
- Report additional metrics (RMSE, MAE, R²) alongside MSE.
- One-hot encode `Segment` and try other models (Gradient Boosting, XGBoost).
- Add a data-drift monitoring job.

---

## Author

**Asrat** — [@aamele](https://github.com/aamele)
