from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

import pandas as pd
import joblib
import re
import string
import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df.sample(frac=1).reset_index(drop=True)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def preprocess_dataframe(df):
    df['text'] = df['text'].apply(preprocess)
    df['title'] = df['title'].apply(preprocess)
    return df

def vectorize_text(X_train, X_test):
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train["title"] + " " + X_train["text"])
    X_test_tfidf = tfidf_vectorizer.transform(X_test["title"] + " " + X_test["text"])
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

def train_and_evaluate_model(X_train_tfidf, y_train, X_test_tfidf, y_test):
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Validation Accuracy: {accuracy:.2f}')
    return model

def save_model(model, vectorizer, model_path, vectorizer_path):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("Model and vectorizer saved successfully.")

if __name__ == "__main__":
    df = load_data("web/ML/csv_train/Combined.csv")
    df = preprocess_dataframe(df)

    X = df[["title", "text"]]
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = vectorize_text(X_train, X_test)

    model = train_and_evaluate_model(X_train_tfidf, y_train, X_test_tfidf, y_test)

    save_model(model, tfidf_vectorizer, 'web/ML/saved_models/logistic_regression_model.pkl', 'web/ML/saved_models/logistic_regression_vectorizer.pkl')