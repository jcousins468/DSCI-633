import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load training data
data_train = pd.read_csv("/Users/jonahcousins/Documents/DSCI_633/DSCI-633/assignments/data/Iris_train.csv")
independent = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
dependent = 'Species'
X = data_train[independent]
Y = data_train[dependent]

# Train model
dtc = DecisionTreeClassifier()
dtc.fit(X, Y)

# Load testing data
data_test = pd.read_csv("/Users/jonahcousins/Documents/DSCI_633/DSCI-633/assignments/data/Iris_test.csv")
X_test = data_test[independent]

# Predict
predictions = dtc.predict(X_test)

# Predict probabilities
probs = dtc.predict_proba(X_test)

# Print a sample row from the training data
print(data_train.iloc[20])

# Print the predictions
print(pd.Series(predictions, name='Species'))

# Print the testing data with predictions
data_test['Species'] = predictions
print(data_test)

# Print the predictions with probabilities
for i, pred in enumerate(predictions):
    print(f"{pred}\t{max(probs[i]):.6f}")