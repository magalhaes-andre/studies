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