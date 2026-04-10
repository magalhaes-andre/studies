"""Feature scaling refresher (with a simple train/test workflow).

Feature scaling transforms numeric input features onto comparable ranges so that a model
(or a distance/gradient-based algorithm) is not unduly influenced by units and magnitude.
This script generates a toy credit-style dataset, encodes a categorical feature, splits
into train/test sets, and applies standardization.

Key ideas used here:
- Label encoding: converts the categorical 'Historical Credit' label into integers so it
  can be used as a numeric feature (note: for non-ordinal categories, one-hot encoding is
  often a better default).
- StandardScaler (z-score): transforms each feature to (x - mean) / std so features have
  approximately zero mean and unit variance.
- Fit on train, transform on both: scaler.fit(x_train) learns mean/std from training data
  only; applying those parameters to x_test avoids leaking information from the test set.

When scaling matters most:
- k-NN, k-means, SVM (RBF), PCA, neural nets, and many regularized linear models.

When it matters less:
- Tree-based models (decision trees, random forests, gradient boosting) are usually
  insensitive to monotonic feature scaling.
"""

import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.preprocessing as pp

np.random.seed(42)

qty_samples = 1000
age = np.random.randint(18, 65, size=qty_samples)
salary = np.random.randint(20000, 150000, size=qty_samples)
historical_credit = np.random.choice(['good', 'bad'], size=qty_samples)
score = np.random.randint(300, 850, size=qty_samples)
qty_loans = np.random.randint(0, 5, size=qty_samples)

target = np.where((age < 30) & (salary > 70000) & (historical_credit == 'good'), 1, 0)

data = pd.DataFrame({
    'Age': age,
    'Salary': salary,
    'Historical Credit': historical_credit,
    'Score': score,
    'Quantity of Loans': qty_loans,
    'Target': target

})

label_encoder = pp.LabelEncoder()
data['Historical Credit'] = label_encoder.fit_transform(data['Historical Credit'])

data_shuffled = data.sample(frac=1, random_state=42)

data_shuffled.to_csv('dataset_credit_shuffled_with_features.csv', index=False)

x = data.drop('Target', axis=1)
y = data['Target']

x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)

x_test_scaled = scaler.transform(x_test)

print(x_test_scaled)

print(x_train_scaled)