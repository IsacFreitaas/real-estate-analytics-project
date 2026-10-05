# Real Estate Analytics Project

<img src="images/real-estate-thumbnail.jpg">

## 1. Project **Description**

* In this project, I have combined **exploratory data analysis** and **machine learning to investigate** the factors associated with **real estate prices** by analyzing the relationship between socioeconomic, demographic, and geographic characteristics and house values, using the California housing dataset as **a case study**.

* The analysis **explores the patterns** present in the dataset and evaluates whether these characteristics can be used to **predict house values**.

* A baseline approach and three machine learning models were evaluated: **Linear Regression**, **Random Forest Regressor**, and an optimized **XGBoost** pipeline.

## 2. Dataset

* This project uses the **California Housing dataset** provided through Scikit-learn. The dataset **contains valuable information** about **housing districts** in California, including socioeconomic, demographic, and geographic characteristics.

* **This dataset is derived from the 1990 U.S. census**, using one row per census block group. The **smallest geographic unit** for which sample data is released by the U.S. Census Bureau is *a block group*, which usually has a population of 600 to 3,000.

* **The target variable** is `MedHouseVal`, representing the **median house value** for each district.

## 3. Project **Objective**

The main objective of this project is to **investigate which characteristics are associated with house values** and evaluate the ability of machine learning models to **predict these values**.

------

This project focuses on **three main questions**:

1. Which **variables** shows the **strongest relationship** with **house values**?
2. Can socioeconomic, demographic, and geographic information be used to **predict house values**?
3. Which of the **evaluated models** provides the **most accurate predictions**?

## 4. Solution **Pipeline**

I have used the **following analytical pipeline** throughout this project:

1. **Load and understand** the dataset.
2. **Explore** the data and **identify relevant patterns**.
3. **Analyze** the relationship between the **available features and house values**.
4. Prepare the data for **machine learning**.
5. Split the data once into **training and test sets**.
6. Establish a **baseline** for model evaluation.
7. **Compare** the models with **cross-validation** on the training set and **tune** the best candidate.
8. Evaluate the **final model once** on the untouched test set.
9. Analyze the prediction errors, interpret the results and discuss the limitations.

**The workflow** that I have implemented in this project **aligns with established data science frameworks** like **CRISP-DM** and **OSEMN**, structured into the **pipeline above**.

The exploratory analysis and the initial modelling are documented in the notebooks. The final cross-validated comparison, the tuning and the error analysis are reproducible through the scripts in `scripts/`.

### Machine learning architecture

```text
Train/test split (80/20, fixed seed)
        ↓
Training set ──► 5-fold cross-validation ──► model comparison and tuning
        ↓
Final pipeline fitted on the training set
        ↓
Untouched test set ──► single final evaluation
```

* **Pipeline:** every model is wrapped in a scikit-learn `Pipeline` that combines a preprocessing step (`ColumnTransformer` with a median imputer for the numeric features) and the estimator. The same object is used for cross-validation, tuning, final evaluation and error analysis.

* **Cross-validation:** the training set is split into 5 folds (`KFold`, shuffled, fixed seed). Each model is trained on 4 folds and scored on the remaining one, and the **mean MAE** across folds is used to compare models.

* **Data leakage:** because preprocessing lives inside the `Pipeline`, it is fitted only on the training folds of each split and never sees the validation fold or the test set. The test set is not used for model selection or tuning; it is used once, at the end.

* **Reproducibility:** the split, the folds and the stochastic models use a single seed (`RANDOM_STATE = 42`, defined in `src/repro.py`).

The reusable code lives in `src/` (`data.py`, `preprocessing.py`, `pipeline.py`, `models.py`, `cv.py`, `tuning.py`, `evaluation.py`, `analysis.py`, `visualization.py`).

## 5. Technologies and Tools

I have used the following **technologies and tools** throughout this project:

- [Python 3.14](https://www.python.org)
- [Pandas](https://pandas.pydata.org) and [NumPy](https://numpy.org)
- [Matplotlib](https://matplotlib.org) and [Seaborn](https://seaborn.pydata.org)
- [Scikit-learn](https://scikit-learn.org/) and [SciPy](https://scipy.org)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Jupyter Notebook](https://jupyter.org)
- [Pytest](https://pytest.org) and [Ruff](https://docs.astral.sh/ruff/)
- [GitHub Actions](https://docs.github.com/actions)

## 6. Main **Insights**

The **exploratory data analysis** that have been performed revealed several **relevant patterns**:

1. **Median income showed the strongest relationship with house values.**

   **Higher-income areas** generally presented **higher house values**, making `MedInc` the variable most strongly associated with the **target**.

<p align="left">
  <img src="images/median-income-vs-house-value.png" width="500">
</p>

2. **Geographic location contains relevant information.**

   House values **are not randomly distributed** across California. The geographic visualization suggests that **areas closer to the coast tend to concentrate higher house values**. However, this relationship was not formally quantified in this project and should therefore be interpreted as just a **interesting visual pattern**.

<p align="left">
  <img src="images/geographic-distribution.png" width="500">
</p>

3. **The dataset contains an upper value limit.**

   A noticeable concentration of observations exists at the **maximum house value** recorded in the dataset (US$500,000). This limitation must be considered when interpreting the distribution of house values and the model predictions.

<p align="left">
  <img src="images/house-value-distribution.png" width="500">
</p>

4. **The relationships between the features and house values are not purely linear.**

   Although some variables showed **clear associations with house values**, the relationships contain patterns that **cannot be fully captured** by a simple linear model. This became more evident during the modelling stage, where Random Forest Regression model **achieved substantially better performance** than Linear Regression model.

## 7. Modelling

* The data was divided into **training and testing sets** to evaluate the models on observations that were not used during training.

* The following approaches were evaluated:

### **Baseline**

* A simple baseline was created by **predicting the average house value** from the training data for every observation in the **test set**.

* This approach provides a reference for determining whether the machine learning models are **learning useful patterns from the available features**.

### **Linear Regression model**

* Linear Regression was used as the first machine learning model because of its simplicity and interpretability.

### **Random Forest Regressor model**

* Random Forest Regressor was used as the second machine learn model to **capture more complex** and potentially non-linear **relationships between the available features and house values**.

### **XGBoost model**

* XGBoost was added as a third model because gradient-boosted trees often perform well on tabular data. With default parameters it already improved on Random Forest (CV MAE 0.316 vs 0.335).

### **Hyperparameter optimization**

* The XGBoost hyperparameters were tuned with **RandomizedSearchCV**, which samples a fixed number of configurations from predefined ranges instead of testing every combination.

* The search used **20 sampled configurations**, each scored with the **5-fold cross-validated MAE** on the **training set only**. Cross-validation estimates how each configuration generalizes without touching the test set.

* The searched ranges are defined in `src/tuning.py` (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha` and `reg_lambda`).

* The best configuration was refitted on the full training set and evaluated once on the test set:

| Hyperparameter | Selected value |
|---|---:|
| `n_estimators` | 307 |
| `max_depth` | 8 |
| `learning_rate` | 0.0476 |
| `min_child_weight` | 5 |
| `subsample` | 0.876 |
| `colsample_bytree` | 0.798 |
| `gamma` | 0.0004 |
| `reg_alpha` | 0.0095 |
| `reg_lambda` | 3.46 |

* These values are the result of a limited random search (20 samples) on this dataset and split. They should not be interpreted as universally optimal, and a different seed or a larger search could select a different configuration.

* The full search summary is saved in `outputs/xgb_random_search.json` and can be reproduced with `python -m scripts.xgb_tuning`.

------

## 8. Results

The models were evaluated using **Mean Absolute Error (MAE)**, where lower values indicate predictions that are, on average, closer to the **actual house values**. The target is expressed in units of US$100,000, so an MAE of 0.29 corresponds to an average error of approximately US$29,000.

Cross-validation (CV) results are the mean MAE over 5 folds of the training set, and test results come from the untouched test set (4,128 observations). Each column must be compared only with itself.

| Model | CV MAE (mean ± std) | Test MAE | Test error (USD) |
|---|---:|---:|---:|
| Baseline (mean of training target) | 0.914 ± 0.010 | 0.906 | ~US$90,600 |
| Linear Regression | 0.529 ± 0.009 | 0.533 | ~US$53,300 |
| Random Forest | 0.335 ± 0.005 | 0.328 | ~US$32,800 |
| XGBoost (default parameters) | 0.316 ± 0.007 | 0.311 | ~US$31,100 |
| **XGBoost (optimized)** | **0.294 ± 0.005** | **0.290** | **~US$29,000** |

The **optimized XGBoost** pipeline achieved the lowest MAE in both cross-validation and the final test evaluation. The CV and test values are close for every model, which suggests the selection process did not overfit the training folds.

<p align="left">
   <img src="images/model-comparison-optimized-xgb.png" width="500">
</p>

The chart shows the cross-validation MAE of each model.

* Compared to the baseline, the optimized XGBoost reduced the test error by approximately **67.99%**.

* Compared to Random Forest, the optimized XGBoost reduced the test error by approximately **11.53%**.

* Compared to the default XGBoost, tuning reduced the test error by approximately **6.66%**.

------

### **Error analysis**

The errors of the optimized XGBoost were analyzed on the test set (details in [docs/error_analysis.md](docs/error_analysis.md); reproducible with `python -m scripts.error_analysis`):

* The residuals (actual − predicted) are centered near zero (mean −0.001), but they are positively skewed: most predictions are close (median absolute error 0.184), while a minority of large under-predictions raises the MAE to 0.290.
* The error **increases with the house value**: MAE is 0.198 for houses valued up to US$100,000 and 0.623 for houses above US$400,000.
* Houses above US$400,000 are 8.2% of the test samples but account for 17.6% of the total absolute error, and the model tends to **under-predict** them.
* Observations at the **target cap** (US$500,000) have an MAE of 0.629, against 0.274 for the remaining observations.

<p align="left">
   <img src="images/error-analysis/actual-vs-predicted.png" width="500">
</p>

The points follow the diagonal, and the horizontal band at the actual value of 5.0 corresponds to the capped observations.

<p align="left">
   <img src="images/error-analysis/error-by-value-range.png" width="700">
</p>

### **Feature importance**

The **feature importance** of the **Random Forest Regressor** (impurity-based, from the modelling notebook) was used to interpret which variables the model relied on. It was not computed for the final XGBoost model.

<p align="left">
  <img src="images/feature-importance.png" width="500">
</p>

* `MedInc` showed the highest importance (about 52.5% of the total), consistent with its strong correlation with `MedHouseVal` in the exploratory analysis.
* `AveOccup` (13.8%), `Latitude` (8.9%) and `Longitude` (8.9%) came next. The two coordinates together contribute about 17.7%, which suggests that location carries information about house values beyond what the linear correlations show.
* Feature importance describes how much the model used each variable, not a causal effect on house prices.

## 9. Limitations and Future Improvements

This project has some **limitations** that should be considered when interpreting the results:

1. **Dataset limitations**

   The dataset represents a specific housing dataset and does not contain all variables that may influence real-world house prices.

2. **Upper value limit**

   The target variable appears **censored at US$500,000** (4.8% of the observations). In the error analysis, these observations had a higher MAE (0.629) and were often under-predicted, so the reported errors for **higher-value houses** should be interpreted with caution.

   Error also grows with the house value, and the model tends to under-predict the most expensive houses.

3. **Simplified modelling process**

   The project still uses a relatively simple feature set and limited feature engineering, so there is room to explore richer predictors.

------

### Possible **future improvements** include:

- Feature engineering.
- Testing **additional regression models**.
- Exploring approaches that handle the **censored target** and improve predictions for high-value houses.
- Using external and more recent real estate datasets.
- Deploy as an API for public use.

## 10. How to Run

The project was developed and validated with **Python 3.14.7**. All dependency versions are pinned:

| Package | Version |
|---|---:|
| pandas | 3.0.5 |
| NumPy | 2.5.2 |
| scikit-learn | 1.9.0 |
| SciPy | 1.18.1 |
| matplotlib | 3.11.1 |
| seaborn | 0.13.2 |
| XGBoost | 3.4.1 |
| notebook | 7.6.3 |
| pytest (development) | 9.1.1 |
| Ruff (development) | 0.16.10 |

### 1. Clone the repository
```bash
git clone https://github.com/IsacFreitaas/real-estate-analytics-project
```

### 2. Navigate to the project directory
```bash
cd real-estate-analytics-project
```

### 3. Create a [virtual environment](https://youtu.be/kyiLBafjpMQ)
```bash
python -m venv .venv
```

### 4. Activate the virtual environment

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install the dependencies

```bash
pip install -r requirements.txt
```

To also install the testing and linting tools, use `requirements-dev.txt` instead (it includes `requirements.txt`):

```bash
pip install -r requirements-dev.txt
```

### 6. Start Jupyter Notebook
```bash
jupyter notebook
```

Then, execute the notebooks **in the order** (`1. EDA.ipynb`, then `2. Modelling.ipynb`) from the `notebooks/` folder.

### 7. Reproduce the final results

Run the scripts from the project root:

```bash
python -m scripts.model_comparison   # CV comparison and final test evaluation
python -m scripts.xgb_tuning         # RandomizedSearchCV for XGBoost (the slowest step)
python -m scripts.error_analysis     # error analysis of the optimized XGBoost
```

Results are written to `outputs/` and the plots to `images/`. The exact versions used to verify reproducibility are also listed in `requirements-locked.txt` (see [docs/reproducibility.md](docs/reproducibility.md)).

### 8. Run the tests and linting

With the virtual environment activated and the development dependencies installed:

```bash
python -m pytest
ruff check .
```

Notebooks are intentionally not linted: reusable and testable code lives in `src/`, while the notebooks hold the exploratory analysis and the project narrative.

## 11. Continuous Integration

The GitHub Actions workflow in `.github/workflows/ci.yml` runs on pushes and on Pull Requests targeting `develop` or `main`:

```text
Pull Request / Push
        ↓
Install dependencies
        ↓
Ruff
        ↓
Pytest
        ↓
PASS / FAIL
```

Introducing automated linting revealed a small number of existing style issues in the project. These were corrected before enabling the CI workflow, ensuring that future changes are automatically checked for the same class of problems.

## 12. Project Structure

```text
real-estate-analytics-project
├── .github/workflows/
│   └── ci.yml
├── docs/
│   ├── error_analysis.md
│   ├── repo_review.md
│   └── reproducibility.md
├── images/
│   ├── error-analysis/
│   └── (EDA, model comparison and thumbnail images)
├── notebooks/
│   ├── 1. EDA.ipynb
│   └── 2. Modelling.ipynb
├── outputs/
│   ├── error_analysis/
│   └── (CV, test and tuning results in JSON)
├── scripts/
│   ├── check_repro.sh
│   ├── cv_demo.py
│   ├── error_analysis.py
│   ├── model_comparison.py
│   ├── pipeline_demo.py
│   └── xgb_tuning.py
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   ├── cv.py
│   ├── data.py
│   ├── evaluation.py
│   ├── models.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── repro.py
│   ├── tuning.py
│   └── visualization.py
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_evaluation.py
│   └── test_pipeline.py
├── .gitattributes
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
├── requirements-dev.txt
└── requirements-locked.txt
```

### `notebooks/`

It contains the Jupyter Notebooks used for the **exploratory data analysis** and the **initial modelling workflow**.

- `1. EDA.ipynb`: Exploratory analysis of the dataset and identification of relevant patterns.
- `2. Modelling.ipynb`: Single-split modelling of the baseline, Linear Regression and Random Forest, with residual analysis and feature importance. The cross-validated comparison, XGBoost tuning and final error analysis are produced by the scripts in `scripts/`.

### `src/`

Contains **reusable Python modules** used throughout the project.

- `data.py`: Dataset loading and train/test split.
- `preprocessing.py` and `pipeline.py`: Preprocessing step and the model `Pipeline` builder.
- `models.py`: Baseline, Linear Regression, Random Forest and XGBoost pipelines.
- `cv.py` and `tuning.py`: Cross-validation and the XGBoost `RandomizedSearchCV`.
- `evaluation.py`: MAE and error reduction.
- `analysis.py`: Residual, value-range and target-cap error analysis.
- `visualization.py`: Reusable plotting functions.
- `repro.py`: Central random seed.

### `scripts/`, `outputs/` and `tests/`

- `scripts/`: Runnable entry points that reproduce the results (`python -m scripts.<name>`).
- `outputs/`: JSON/CSV results generated by the scripts.
- `tests/`: Pytest tests for data validation, MAE and the model pipelines.

## 13. [Get in touch](https://linktr.ee/isaczeitgeistpy)

* LinkedIn: https://www.linkedin.com/in/isac-freitas-16a035223/
* Gmail: isaczeitgeist+contatogithub@gmail.com

<p>
  <img src="images/Isac-Freitas-LinkedIn-Banner.jpg">
</p>