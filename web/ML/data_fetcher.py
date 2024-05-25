import joblib
import re
import string
import polars as pl
from scipy.special import expit
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
saved_models_dir = os.path.join(current_dir, "saved_models")

def process_data():
    df1 = pl.read_csv(os.path.join(current_dir, "csv_train/Fake.csv"), separator=',')
    df2 = pl.read_csv(os.path.join(current_dir, "csv_train/True.csv"), separator=',')

    df1 = df1.with_columns(
        pl.lit(0).alias("Label")
    )

    df2 = df2.with_columns(
        pl.lit(1).alias('Label')
    )

    combine_df = pl.concat([df1, df2]).drop(["date", "subject"])

    combine_df.write_csv(os.path.join(current_dir, 'csv_train/Combined.csv'))

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def load_model_and_vectorizer(model_name: str):
    model_path = os.path.join(saved_models_dir, f"{model_name}_model.pkl")
    vectorizer_path = os.path.join(saved_models_dir, f"{model_name}_vectorizer.pkl")

    print(f"Loading {model_name.replace('_', ' ').title()} model from {model_path}")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer

def classify_article(model, vectorizer, title: str, text: str):
    combined = title + " " + text
    processed_text = preprocess(combined)
    combined_vectorized = vectorizer.transform([processed_text])
    pred_proba = model.predict_proba(combined_vectorized)

    proba_real = pred_proba[0][1] * 100
    return proba_real

def fetch_prediction_random_forest(title: str, text: str) -> float:
    model, vectorizer = load_model_and_vectorizer("random_forest")
    return classify_article(model, vectorizer, title, text)

def fetch_prediction_decision_tree(title: str, text: str) -> float:
    model, vectorizer = load_model_and_vectorizer("decision_tree")
    return classify_article(model, vectorizer, title, text)

def fetch_prediction_logistic_regression(title: str, text: str) -> float:
    model, vectorizer = load_model_and_vectorizer("logistic_regression")
    return classify_article(model, vectorizer, title, text)

def fetch_prediction_multinomial_nb(title: str, text: str) -> float:
    model, vectorizer = load_model_and_vectorizer("multinomial_NB")
    return classify_article(model, vectorizer, title, text)

def fetch_prediction_svm(title: str, text: str) -> float:
    model, vectorizer = load_model_and_vectorizer("svm")

    combined = title + " " + text
    processed_text = preprocess(combined)
    combined_vectorized = vectorizer.transform([processed_text])
    decision_function = model.decision_function(combined_vectorized)
    proba = expit(decision_function)

    proba_real = proba[0] * 100
    return float(f'{proba_real:.2f}')