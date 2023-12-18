import pandas as pd
from sklearn.linear_model import Lasso
import numpy as np
import spec_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/dataset.csv")
X_columns = spec_utils.get_wavelengths(True)
y_column = "oc"
df = df[X_columns + [y_column]]

scalers = {}

for column in df.columns:
    scaler = MinMaxScaler()
    column_data = df[column].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(column_data)
    scalers[column] = scaler
    df[column] = scaled_data.flatten()

X = df[X_columns]
y = df[y_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)
lasso = Lasso(alpha=0.001)
lasso.fit(X_train,y_train)
indices = np.argsort(np.abs(lasso.coef_))[::-1][:2]

for i in range(len(lasso.coef_)):
    if i not in indices:
        lasso.coef_[i] = 0
print(lasso.coef_)
print(lasso.score(X_test,y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)
lasso = Lasso(alpha=0.001)
lasso.fit(X_train,y_train)
indices = np.argsort(np.abs(lasso.coef_))[::-1][:10]

for i in range(len(lasso.coef_)):
    if i not in indices:
        lasso.coef_[i] = 0

print(lasso.coef_)
print(lasso.score(X_test,y_test))

