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