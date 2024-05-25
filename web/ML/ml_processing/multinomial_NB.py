from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import pandas as pd
import joblib
import re
import string

def load_data(filepath):
    df = pd.read_csv(filepath)
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
    tfidf_vectorizer = TfidfVectorizer()
    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)
    return x_train_tfidf, x_test_tfidf, tfidf_vectorizer

def train_and_evaluate_model(x_train_tfidf, y_train, x_test_tfidf, y_test):
    model = MultinomialNB()
    model.fit(x_train_tfidf, y_train)
    y_pred = model.predict(x_test_tfidf)
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
    
    x = df["title"] + " " + df["text"]
    y = df["Label"]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    x_train_tfidf, x_test_tfidf, tfidf_vectorizer = vectorize_text(x_train, x_test)
    
    model = train_and_evaluate_model(x_train_tfidf, y_train, x_test_tfidf, y_test)
    
    save_model(model, tfidf_vectorizer, 'web/ML/saved_models/multinomial_NB_model.pkl', 'web/ML/saved_models/multinomial_NB_vectorizer.pkl')