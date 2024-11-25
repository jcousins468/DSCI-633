import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from gensim.parsing.preprocessing import remove_stopwords, strip_tags
import sys
sys.path.append('/Users/jonahcousins/Documents/DSCI_633/DSCI-633/assignments')
from Evaluation.my_evaluation import my_evaluation
from Tuning.my_GA import my_GA

class my_model():
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.clf = RandomForestClassifier()
    
    def preprocess_text(self, text):
        text = strip_tags(text)
        return remove_stopwords(text.lower())
    
    def obj_func(self, predictions, actuals, pred_proba=None):
        # One objectives: higher f1 score
        eval = my_evaluation(predictions, actuals, pred_proba)
        return eval.f1()

    def fit(self, X, y):
        # Validate input
        if 'description' not in X.columns:
            raise ValueError("The input data must contain a 'description' column.")
        # Process descriptions
        X['processed_description'] = X['description'].apply(self.preprocess_text)
        # Vectorize the text
        X_vectorized = self.vectorizer.fit_transform(X['processed_description'])
        X_vectorized = pd.DataFrame(X_vectorized.toarray())
        
        # Tune the model using genetic algorithm
        ga = my_GA(RandomForestClassifier, X.copy(), y, 
           {"n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]},
           self.obj_func, 
           generation_size=50, crossval_fold=5, max_generation=10, max_life=2)
        
        best = ga.tune()[0]
        dec_dict = {key: best[i] for i, key in enumerate(["loss", "penalty", "alpha"])}
        self.clf = SGDClassifier(**dec_dict)
        self.clf.fit(X_vectorized, y)


    def predict(self, X):
        # Validate input
        if 'description' not in X.columns:
            raise ValueError("The input data must contain a 'description' column.")
        # Process descriptions
        X['processed_description'] = X['description'].apply(self.preprocess_text)
        # Vectorize using the same vectorizer as fit
        X_vectorized = self.vectorizer.transform(X['processed_description'])
        predictions = self.clf.predict(X_vectorized)
        return predictions