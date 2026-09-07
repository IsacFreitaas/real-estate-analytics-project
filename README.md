# Real Estate Analytics Project

<img src="images/real-estate-thumbnail.jpg">

------

## 1. Description

This project combines exploratory data analysis and machine learning to investigate the relationship between socioeconomic, demographic, and geographic characteristics and house values in California.

The analysis explores the patterns present in the dataset and evaluates whether these characteristics can be used to predict house values.

A baseline approach and two machine learning models were evaluated: Linear Regression and Random Forest Regressor.

------

## 2. Technologies and Tools

The following technologies and tools were used throughout this project:

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Git and GitHub

------

## 3. Project Objective

The main objective of this project is to investigate which characteristics are associated with house values and evaluate the ability of machine learning models to predict these values.

The project focuses on three main questions:

1. Which variables show the strongest relationship with house values?
2. Can socioeconomic, demographic, and geographic information be used to predict house values?
3. Which of the evaluated models provides the most accurate predictions?

------

## 4. Solution Pipeline

The following analytical pipeline was used throughout this project:

1. Load and understand the dataset.
2. Explore the data and identify relevant patterns.
3. Analyze the relationship between the available features and house values.
4. Prepare the data for machine learning.
5. Split the data into training and testing sets.
6. Establish a baseline for model evaluation.
7. Train and compare machine learning models.
8. Evaluate the best-performing model.
9. Interpret the results and discuss the limitations.

Each stage is documented in detail throughout the project's notebooks.

------

## 5. Main Insights

The exploratory data analysis revealed several relevant patterns:

1. **Median income showed the strongest relationship with house values.**

   Higher-income areas generally presented higher house values, making `MedInc` the variable most strongly associated with the target.

2. **Geographic location also contains relevant information.**

   The geographic distribution showed that house values are not randomly distributed across California. The visualization also suggests that areas closer to the coast tend to concentrate higher house values.

3. **The dataset contains an upper value limit.**

   A noticeable concentration of observations exists at the maximum house value recorded in the dataset. This pattern was also relevant when interpreting the model predictions.

4. **The relationships between the features and house values are not purely linear.**

   Although some variables show clear associations with house values, the relationships contain pat
   
------

   ## 6. Modelling

The data was divided into training and testing sets to evaluate the models on observations that were not used during training.

The following approaches were evaluated:

### Baseline

A simple baseline was created by predicting the average house value from the training data for every observation in the test set.

This approach provides a reference for determining whether the machine learning models are learning useful patterns from the available features.

### Linear Regression

Linear Regression was used as the first machine learning model because of its simplicity and interpretability.

### Random Forest Regressor

Random Forest Regressor was used to capture more complex and potentially non-linear relationships between the available features and house values.

------

## 7. Results

The models were evaluated using Mean Absolute Error (MAE), where lower values indicate predictions that are, on average, closer to the actual house values.

| Model | MAE | Average Error |
|---|---:|---:|
| Baseline | 0.906 | US$90,607 |
| Linear Regression | 0.533 | US$53,320 |
| Random Forest | 0.328 | US$32,784 |

The Random Forest achieved the best performance among the evaluated approaches.

Compared to the baseline, the Random Forest reduced the prediction error by approximately **63.82%**.

Compared to Linear Regression, it achieved an additional error reduction of approximately **38.51%**.

------

The feature importance analysis showed that `MedInc` was the most influential variable in the Random Forest model, reinforcing the relationship identified during the exploratory data analysis.

`AveOccup`, `Latitude`, and `Longitude` also contributed relevant information to the predictions.

------

## 8. Limitations and Future Improvements

This project has some limitations that should be considered when interpreting the results:

1. **Dataset limitations**

   The dataset represents a specific housing dataset and does not contain all variables that may influence real-world house prices.

2. **Upper value limit**

   The target variable contains a noticeable upper limit, which may affect the model's ability to accurately predict some higher-value observations.

3. **Simplified modelling process**

   The models were evaluated using a relatively simple modelling workflow without extensive hyperparameter tuning or advanced feature engineering.

------

Possible future improvements include:

- Hyperparameter tuning.
- Cross-validation.
- Feature engineering.
- Testing additional regression models.
- Using external and more recent real estate datasets.
API
***

------

## 9. Project Structure

```text
├── notebooks/
│   ├── 1. EDA.ipynb
│   └── 2. Modelling.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluation.py
│   └── visualization.py
│
├── images/
│   └── real-estate-thumbnail.jpg
│
├── README.md
├── requirements.txt
└── .gitignore
```

### `notebooks/`

Contains the Jupyter Notebooks used for the exploratory data analysis and machine learning workflow.

- `1. EDA.ipynb`: Exploratory analysis of the dataset and identification of relevant patterns.
- `2. Modelling.ipynb`: Data preparation, model training, evaluation, and interpretation of the results.

### `src/`

Contains reusable Python modules used throughout the project.

- `data.py`: Dataset loading functions.
- `evaluation.py`: Functions for evaluating model performance.
- `visualization.py`: Reusable visualization functions used during model evaluation.

### `images/`

Contains images used in the project documentation.

------

## 10. How to Run

### 1. Clone the repository
```bash
git clone https://github.com/IsacFreitaas/real-state-analytics-project
```

### 2. Navigate to the project directory
```bash
cd real-state-analytics-project
```

### 3. Create a virtual environment
```bash
python -m venv venv
```

### 4. Activate the virtual environment

macOS/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### 5. Install the dependencies

```bash
pip install -r requirements.txt
```

### 6. Start Jupyter Notebook
```bash
jupyter notebook
```

Then, execute the notebooks in the following order:

1. `1. EDA.ipynb`
2. `2. Modelling.ipynb`


---

# 11. Dataset

This project uses the California Housing dataset provided through Scikit-learn.

The dataset contains information about housing districts in California, including socioeconomic, demographic, and geographic characteristics.

The target variable is `MedHouseVal`, representing the median house value for each district.


imagens:
Uma imagem de capa.
Um gráfico geográfico.
O gráfico de comparação dos modelos.
Talvez o gráfico de feature importance.

Descrição
   ↓
Insights
   ↓
[Mapa geográfico]
   ↓
Modelagem
   ↓
[Comparação dos modelos]
   ↓
Resultados