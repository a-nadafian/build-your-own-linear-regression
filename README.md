# Build Your Own Linear Regression

This project implements Linear Regression from scratch using gradient descent, alongside a head-to-head comparison with scikit-learn's `LinearRegression`. It is modular, easy to read, and designed for learning and experimentation.

## Project Structure

```
build-your-own-linear-regression/
├── data/
│   ├── raw/Salary_Data.csv
│   └── processed/processed_salary_data.csv
├── notebooks/
│   ├── 01-initial-exploration.ipynb     # Data cleaning + EDA
│   └── 02-model-validation.ipynb        # Compare custom vs sklearn
├── src/linear_regression_model/
│   ├── __init__.py                      # Exposes LinearRegressionGD
│   ├── model.py                         # Model with fit/predict/score
│   └── utils.py                         # Cost fn + gradient helpers
└── requirements.txt
```

## Dataset

The dataset is prepared in `notebooks/01-initial-exploration.ipynb`. We standardize categorical fields, remove duplicates, handle missing values, and save the processed data to `data/processed/processed_salary_data.csv`. The primary signal used for the baseline model is `Years of Experience` to predict `Salary`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

## Usage

Example training with the custom gradient-descent model:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from linear_regression_model import LinearRegressionGD

df = pd.read_csv('data/processed/processed_salary_data.csv')
X = df[["Years of Experience"]].values
y = df["Salary"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegressionGD(learning_rate=0.005, epochs=2000, fit_intercept=True, random_state=42)
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("R2 (train):", model.score(X_train, y_train))
print("R2 (test):", model.score(X_test, y_test))
```

## Validation Notebook

Open and run `notebooks/02-model-validation.ipynb`. It trains both the custom model and scikit-learn's model, compares coefficients and metrics (MSE, R^2), and plots the regression lines.

## Development

- Model API: `fit(X, y)`, `predict(X)`, `score(X, y)`
- Utilities in `src/linear_regression_model/utils.py` implement MSE, R^2, and gradient descent.

## License

MIT License. See `LICENSE`.
