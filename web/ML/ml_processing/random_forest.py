from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import numpy as np
import re
import string
import joblib

def load_data(combined_path):
    df = pd.read_csv(combined_path)
    return df.sample(frac=1).reset_index(drop=True)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
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

def vectorize_text(x_train, x_test):
    vectorization = TfidfVectorizer()
    x_train_tfidf = vectorization.fit_transform(x_train)
    x_test_tfidf = vectorization.transform(x_test)
    return x_train_tfidf, x_test_tfidf, vectorization

def train_and_evaluate_model(x_train_tfidf, y_train, x_test_tfidf, y_test):
    model = RandomForestClassifier(random_state=0)
    model.fit(x_train_tfidf, y_train)
    accuracy = model.score(x_test_tfidf, y_test)
    print(f'Validation Accuracy: {accuracy:.2f}')
    return model

def save_model(model, vectorizer, model_path, vectorizer_path):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("Model and vectorizer saved successfully.")

if __name__ == "__main__":
    df = load_data("web/ML/csv_train/Combined.csv")
    df = preprocess_dataframe(df)
    
    x = df["title"] + " " + df["text"]
    y = df["Label"]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    x_train_tfidf, x_test_tfidf, vectorization = vectorize_text(x_train, x_test)
    
    model = train_and_evaluate_model(x_train_tfidf, y_train, x_test_tfidf, y_test)
    
    save_model(model, vectorization, 'web/ML/saved_models/random_forest_model.pkl', 'web/ML/saved_models/random_forest_vectorizer.pkl')