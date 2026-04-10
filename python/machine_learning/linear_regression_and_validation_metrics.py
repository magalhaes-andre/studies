"""Linear regression + basic error metrics refresher.

Linear regression models a numeric target as a linear combination of input features.
This script fits an Ordinary Least Squares (OLS) regression with statsmodels on a small
toy dataset where ice-cream units sold depend on temperature and marketing spend.

Key ideas used here:
    - OLS objective: chooses coefficients that minimize the sum of squared residuals
    (errors) between observed and predicted values.
    - Intercept term: statsmodels OLS does not automatically add a bias column; adding a
    constant column (here named 'Intercepto') allows the model to learn a baseline level
    of sales when other features are zero.
    - Model summary: statsmodels provides coefficient estimates, standard errors, t-stats,
    p-values, R², and diagnostic information useful for interpretation.

Validation metrics computed on the fitted data:
- MAE (mean absolute error): average absolute deviation; robust and in target units.
- MSE (mean squared error): squares errors; penalizes large mistakes more heavily.
- RMSE (root MSE): square root of MSE; back in target units and easier to interpret.

Note:
    - Proper validation typically uses a holdout set or cross-validation rather than
    evaluating on the same data used to fit the model.
"""

import statsmodels.api as sm
import pandas as pd

# Fictional data frame
ice_cream_selling_data = {
    'Units': [200, 300, 400, 350, 500],
    'Temperature': [28, 30, 32, 29, 33],
    'Marketing_Promotion': [1000, 1200, 800, 900, 1100]
}

df = pd.DataFrame(ice_cream_selling_data)

# Intercepto constant TODO: what?

df['Intercepto'] = 1

# Define independant variables (X)
X = df[['Intercepto', 'Temperature', 'Marketing_Promotion']]

# Define dependant variable (Y)
Y = df['Units']

model = sm.OLS(Y, X).fit()

print(model.summary())

## Now to calculate RMSE and MSE
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Predictions
y_prediction = model.predict(X)

mae = mean_absolute_error(Y, y_prediction) # MEAN DEVIATION in units selled 
mse = mean_squared_error(Y, y_prediction) # Lower MSE == more precision
rmse = np.sqrt(mse) # Intuitive interpretation, 39 MEAN DEVIATION in units selled

metrics = {
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse
}

print(metrics)