import pandas as pd

df = pd.read_csv('C:\\Users\\marco\\Desktop\\codes\\KaggleCompetitions\\Fraud\\Fraud.csv')

detected_and_was = ((df['isFlaggedFraud'] == 1) & (df['isFraud'] == 1)).sum()
detected_and_was_not = ((df['isFlaggedFraud'] == 1) & (df['isFraud'] == 0)).sum()
not_detected_and_was = ((df['isFlaggedFraud'] == 0) & (df['isFraud'] == 1)).sum()
not_detected_and_was_not = ((df['isFlaggedFraud'] == 0) & (df['isFraud'] == 0)).sum()

print("Detected fraud and it was indeed fraud:", detected_and_was)
print("Detected fraud but it was not fraud:", detected_and_was_not)
print("Did not detect and it was fraud:", not_detected_and_was)
print("Did not detect and it was not fraud:", not_detected_and_was_not)

print("Correct detections:", detected_and_was)
print("Errors:", detected_and_was_not + not_detected_and_was)