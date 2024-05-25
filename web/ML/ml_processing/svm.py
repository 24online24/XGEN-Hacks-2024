from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC

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
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def preprocess_dataframe(df):
    df['text'] = df['text'].apply(preprocess)
    return df

def vectorize_text(texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f'Validation Accuracy: {accuracy:.2f}')
    print(f'Classification Report:\n{report}')
    return model

def save_model(model, vectorizer, model_path, vectorizer_path):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("Model and vectorizer saved successfully.")

if __name__ == "__main__":
    df = load_data("web/ML/csv_train/Combined.csv")
    df = preprocess_dataframe(df)
    
    texts = df['text'].tolist()
    labels = df['Label'].tolist()
    labels = [int(label) for label in labels]
    
    X, vectorizer = vectorize_text(texts)
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    model = train_and_evaluate_model(X_train, y_train, X_test, y_test)
    
    save_model(model, vectorizer, 'web/ML/saved_models/svm_model.pkl', 'web/ML/saved_models/svm_vectorizer.pkl')