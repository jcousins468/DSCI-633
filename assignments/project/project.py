import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import sys
sys.path.append('/Users/jonahcousins/Documents/DSCI_633/DSCI-633/assignments')
from Tuning.my_GA import my_GA

class my_model:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def preprocess_text(self, text):
        text = text.replace('<', ' ').replace('>', ' ')
        text = ''.join([char if char.isalpha() else ' ' for char in text])
        text = text.lower()
        stop_words = set(["a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "on", "in", "with", "as", "by", "at", "from", "up", "down", "to", "of", "it", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must"])
        words = text.split()
        words = [w for w in words if not w in stop_words]
        return " ".join(words)

    def obj_func(self, model, X, y):
        skf = StratifiedKFold(n_splits=3)  #Number of folds
        scores = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            if 'description' not in X_train.columns:
                raise ValueError("The input data must contain a 'description' column.")
            if 'processed_description' not in X_train.columns:
                X_train['processed_description'] = X_train['description'].apply(self.preprocess_text)
            if 'processed_description' not in X_test.columns:
                X_test['processed_description'] = X_test['description'].apply(self.preprocess_text)
            combined_data = pd.concat([X_train['processed_description'], X_test['processed_description']])
            self.vectorizer.fit(combined_data)
            X_train_vectorized = self.vectorizer.transform(X_train['processed_description'])
            X_test_vectorized = self.vectorizer.transform(X_test['processed_description'])
            model.fit(X_train_vectorized, y_train)
            predictions = model.predict(X_test_vectorized)
            scores.append(f1_score(y_test, predictions))
        return np.mean(scores)

    def fit(self, X, y):
        if 'description' not in X.columns:
            raise ValueError("The input data must contain a 'description' column.")
        X['processed_description'] = X['description'].apply(self.preprocess_text)
        self.vectorizer.fit(X['processed_description'])
        X_vectorized = self.vectorizer.transform(X['processed_description'])
        X_vectorized = pd.DataFrame(X_vectorized.toarray())

        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [3, 6],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0]
        }

        decision_boundary = {
            "n_estimators": (50, 100),
            "max_depth": (3, 6),
            "learning_rate": (0.01, 0.1),
            "subsample": (0.8, 1.0)
        }

        ga = my_GA(param_grid, GradientBoostingClassifier, X.copy(), y, decision_boundary, self.obj_func, generation_size=10, selection_rate=0.5, mutation_rate=0.01, crossval_fold=3, max_generation=3, max_life=1)
        ga.run()
        self.best_params_ = ga.best_params_
        self.clf = GradientBoostingClassifier(**self.best_params_)
        self.clf.fit(X_vectorized, y)

    def predict(self, X):
        if 'description' not in X.columns:
            raise ValueError("The input data must contain a 'description' column.")
        X['processed_description'] = X['description'].apply(self.preprocess_text)
        X_vectorized = self.vectorizer.transform(X['processed_description'])
        predictions = self.clf.predict(X_vectorized)
        return predictions