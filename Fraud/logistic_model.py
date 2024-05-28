import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('Fraud.csv')

cols_to_use = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
X = data[cols_to_use]
y = data.isFraud

X_train, X_valid, y_train, y_valid = train_test_split(X, y)

s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

log_reg = LogisticRegression()
log_reg.fit(OH_X_train, y_train)

y_pred = log_reg.predict(OH_X_valid)
confusion = confusion_matrix(y_valid, y_pred)
accuracy = accuracy_score(y_valid, y_pred)
mae = mean_absolute_error(y_valid, y_pred)

print("Matriz de Confusão:")
print(confusion)
print("Acurácia:")
print(accuracy)
print("Erro Absoluto Médio (MAE):")
print(mae)
