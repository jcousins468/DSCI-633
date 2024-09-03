from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

if __name__ == "__main__":
    #  Load training data
    data_train = pd.read_csv("/Users/jonahcousins/Documents/DSCI_633/DSCI-633/assignments/data/Iris_train.csv")
    
    # Explore the loaded pandas dataframe
    # Print out the 12th training data point
    print(data_train.loc[12])
    # Print out the column "Species"
    print(data_train["SepalWidthCm"] < 2.5)
    # Print out the data points with "Species" == "Iris-setosa"
    print(data_train[data_train["Species"]=="Iris-setosa"])

    # Separate independent variables and dependent variables
    independent = ["SepalLengthCm",	"SepalWidthCm",	"PetalLengthCm", "PetalWidthCm"]
    X = data_train[independent]
    Y = data_train["Species"]
    # Train model

    dtc = DecisionTreeClassifier()
    dtc.fit(X,Y)
    # Load testing data
    data_test = pd.read_csv("/Users/jonahcousins/Documents/DSCI_633/DSCI-633/assignments/data/Iris_test.csv")
    X_test = data_test[independent]
    # Predict
    predictions = dtc.predict(X_test)
    # Predict probabilities
    probs = dtc.predict_proba(X_test)
    # Print results
    for i,pred in enumerate(predictions):
        print("%s\t%f" %(pred,max(probs[i])))