import pandas as pd

# Carregar o dataset
df = pd.read_csv('C:\\Users\\marco\\Desktop\\codes\\KaggleCompetitions\\Fraud\\Fraud.csv')  # Substitua "seu_dataset.csv" pelo nome do seu arquivo CSV

print("Primeiras 5 linhas do DataFrame:")
print(df.head())

# Mostrar a quantidade de linhas e colunas
linhas, colunas = df.shape
print(f"O DataFrame tem {linhas} linhas e {colunas} colunas.")

# Filtrar apenas transações fraudulentas
fraudulent_transactions = df[df['isFraud'] == 1]

# Contar quantas fraudes cada cliente tem individualmente
fraudulent_transactions_per_customer = fraudulent_transactions.groupby('nameOrig').size()

# Filtrar clientes com mais de uma transação fraudulenta
customers_with_multiple_fraudulent_transactions = fraudulent_transactions_per_customer[fraudulent_transactions_per_customer > 1]

# Imprimir o número de clientes com mais de uma transação fraudulenta
num_customers_with_multiple_fraudulent_transactions = len(customers_with_multiple_fraudulent_transactions)
print("Number of customers with more than one fraudulent transaction:", num_customers_with_multiple_fraudulent_transactions)
