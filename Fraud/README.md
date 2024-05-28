# Kaggle Fraud Detection Competition

## Português

### Descrição do Projeto

Este projeto foi desenvolvido para uma competição no Kaggle, utilizando um modelo de predição para detectar fraudes em transações financeiras. Os dados do caso estão disponíveis em formato CSV com 6.362.620 linhas e 10 colunas.

### Dicionário de Dados

- **step**: Mapeia uma unidade de tempo no mundo real. Neste caso, 1 passo equivale a 1 hora. Total de passos: 744 (simulação de 30 dias).
- **type**: Tipo de transação - CASH-IN, CASH-OUT, DEBIT, PAYMENT e TRANSFER.
- **amount**: Valor da transação em moeda local.
- **nameOrig**: Cliente que iniciou a transação.
- **oldbalanceOrg**: Saldo inicial antes da transação.
- **newbalanceOrig**: Novo saldo após a transação.
- **nameDest**: Cliente que é o destinatário da transação.
- **oldbalanceDest**: Saldo inicial do destinatário antes da transação. Não há informações para clientes que iniciam com M (Merchants).
- **newbalanceDest**: Novo saldo do destinatário após a transação. Não há informações para clientes que iniciam com M (Merchants).
- **isFraud**: Indica se a transação é fraudulenta. Os agentes fraudulentos visam lucrar assumindo o controle das contas dos clientes e tentando esvaziar os fundos transferindo para outra conta e depois sacando do sistema.
- **isFlaggedFraud**: Sinaliza tentativas ilegais de transferência de mais de 200.000 em uma única transação.

## Arquivos no Repositório

- `fraud.csv`: Dataset obtido em: https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data
- `test_system.py`: Análise de 'isFlaggedFraud'.
- `clients.py`: Análise de clientes.
- `my_model.py`: Modelo inicial usando XGBRegressor
- `logistic_model`: modelo utilizando LogisticRegression.
- `gpt_model.py`: Modelo aprimorado com a ajuda do ChatGPT.

### Passos da Análise

1. **Primeira Análise**: Verifiquei o cabeçalho e a quantidade de colunas e linhas do DataFrame, totalizando 6.362.620 linhas e 11 colunas.
2. **Verificação de Data Leakage**: Verifiquei se havia dados faltantes no DataFrame e, felizmente, não havia nenhum, poupando assim tempo e trabalho.
3. **Análise Aprofundada**: Após algum tempo de análise, decidi quais colunas seriam relevantes e quais seriam descartadas. Notei que os dados de clientes não tinham correlação com as fraudes, pois nenhum cliente tinha mais de uma fraude. Então, descartei os dados relacionados a clientes e também 'isFlaggedFraud', que detectou apenas 16 fraudes em 8.213. Tratei as variáveis categóricas usando one-hot encoding.
4. **Modelos**: Após finalizar as análises, criei modelos utilizando o XGBRegressor. Após mais pesquisas, utilizei também LogisticRegression para comparar o desempenho. Pedi ajuda ao ChatGPT para melhorar o modelo, resultando no 'gpt_model.py'. Descobri que o XGBRegressor não era ideal para treinar o modelo e que era incorreto medir a precisão com Mean Absolute Error.

### Conclusão

O melhor modelo foi o XGBClassifier em 'gpt_model.py', ideal para problemas de classificação binária. Considerando ser minha primeira vez estudando e treinando um modelo, foi um bom aprendizado. Pretendo melhorar a precisão do modelo futuramente, estudando mais e atualizando minhas habilidades.

---

## English

### Project Description

This project was developed for a Kaggle competition, using a prediction model to detect fraud in financial transactions. The data is available in CSV format with 6,362,620 rows and 10 columns.

### Data Dictionary

- **step**: Maps a unit of time in the real world. In this case, 1 step equals 1 hour. Total steps: 744 (30-day simulation).
- **type**: Transaction type - CASH-IN, CASH-OUT, DEBIT, PAYMENT, and TRANSFER.
- **amount**: Transaction amount in local currency.
- **nameOrig**: Customer who initiated the transaction.
- **oldbalanceOrg**: Initial balance before the transaction.
- **newbalanceOrig**: New balance after the transaction.
- **nameDest**: Customer who is the transaction recipient.
- **oldbalanceDest**: Initial balance of the recipient before the transaction. No information for customers starting with M (Merchants).
- **newbalanceDest**: New balance of the recipient after the transaction. No information for customers starting with M (Merchants).
- **isFraud**: Indicates if the transaction is fraudulent. Fraudulent agents aim to profit by taking control of customer accounts and trying to empty funds by transferring to another account and then withdrawing from the system.
- **isFlaggedFraud**: Flags illegal transfer attempts over 200,000 in a single transaction.

## Files in the Repository

- `fraud.csv`: Dataset obtained from: https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data
- `test_system.py`: Analysis of 'isFlaggedFraud'.
- `clients.py`: Client analysis.
- `my_model.py`: Initial model using XGBRegressor.
- `logistic_model`: Model using LogisticRegression.
- `gpt_model.py`: Enhanced model with the help of ChatGPT.

### Analysis Steps

1. **First Analysis**: Checked the header and the number of columns and rows of the DataFrame, totaling 6,362,620 rows and 11 columns.
2. **Data Leakage Check**: Checked for missing data in the DataFrame and fortunately, there was none, saving time and effort.
3. **In-Depth Analysis**: After some analysis, I decided which columns were relevant and which to discard. I noticed that customer data had no correlation with fraud since no customer had more than one fraud. So, I discarded customer-related data and also 'isFlaggedFraud', which detected only 16 frauds out of 8,213. I handled categorical variables using one-hot encoding.
4. **Models**: After completing the analysis, I created models using XGBRegressor. After further research, I also used LogisticRegression to compare performance. I asked ChatGPT to improve the model, resulting in 'gpt_model.py'. I learned that XGBRegressor was not ideal for training the model and it was incorrect to measure accuracy with Mean Absolute Error.

### Conclusion

The best model was the XGBClassifier in 'gpt_model.py', ideal for binary classification problems. Considering this was my first time studying and training a model, it was a good learning experience. I plan to improve the model's accuracy in the future, studying more and updating my skills.



