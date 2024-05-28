import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, roc_auc_score, precision_score, recall_score, f1_score

# Carregar os dados
data = pd.read_csv('Fraud.csv')

# Selecionar as colunas para uso
cols_to_use = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
X = data[cols_to_use]
y = data.isFraud

# Dividir os dados em conjuntos de treinamento e validação
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Identificar colunas categóricas
categorical_cols = ['type']

# Aplicar OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[categorical_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[categorical_cols]))

# Alinhar índices
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remover colunas categóricas do conjunto original e adicionar colunas codificadas
num_X_train = X_train.drop(categorical_cols, axis=1)
num_X_valid = X_valid.drop(categorical_cols, axis=1)

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ajustar tipos de colunas
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

# Treinar o modelo XGBoost
model = XGBClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=4)
model.fit(OH_X_train, y_train, early_stopping_rounds=5, eval_set=[(OH_X_valid, y_valid)], verbose=False)

# Fazer previsões e calcular métricas
y_pred = model.predict(OH_X_valid)
mae = mean_absolute_error(y_valid, y_pred)

# Calcular e imprimir métricas de classificação
auc_roc = roc_auc_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred)
recall = recall_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"AUC-ROC: {auc_roc}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
