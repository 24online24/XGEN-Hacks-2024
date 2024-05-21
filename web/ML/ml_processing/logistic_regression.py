import time
import pandas as pd
import numpy as np
import joblib
import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("ML/csv_train/Combined.csv")

df = df.sample(frac=1).reset_index(drop=True)

X = df[["title", "text"]]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(
    X_train["title"] + " " + X_train["text"])
X_test_tfidf = tfidf_vectorizer.transform(
    X_test["title"] + " " + X_test["text"])

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

joblib.dump(model, 'web/ML/saved_models/logistic_regression_model.pkl')
joblib.dump(tfidf_vectorizer,
            'web/ML/saved_models/logistic_regression_vectorizer.pkl')
