import joblib
import pandas as pd
import re
import string
import polars as pl
from scipy.special import expit


def process_data():
    df1 = pl.read_csv("ML/csv_train/Fake.csv", separator=',')
    df2 = pl.read_csv("ML/csv_train/True.csv", separator=',')

    df1 = df1.with_columns(
        pl.lit(0).alias("Label")
    )

    df2 = df2.with_columns(
        pl.lit(1).alias('Label')
    )

    combine_df = pl.concat([df1, df2]).drop(["date", "subject"])

    combine_df.write_csv('ML/csv_train/Combined.csv')


def fetch_prediction_random_forest(title: str, text: str) -> float:
    Random_Forest_Classifier = joblib.load(
        "web/ML/saved_models/random_forest_model.pkl")
    vectorizer = joblib.load(
        'web/ML/saved_models/random_forest_vectorizer.pkl')

    def preprocess(text):
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r"\\W", " ", text)
        text = re.sub(r'https://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        return text

    def Fake_news_predict(news):
        testing_news = {"text": [news]}
        new_df_test = pd.DataFrame(testing_news)
        new_df_test["text"] = new_df_test["text"].apply(preprocess)
        new_x_test = vectorizer.transform(new_df_test["text"])
        pred_proba_RFC = Random_Forest_Classifier.predict_proba(new_x_test)

        proba_real = pred_proba_RFC[0][0] * 100

        return float(f'{proba_real:.2f}')

    return Fake_news_predict(title + " " + text)


def fetch_prediction_decision_tree(title: str, text: str) -> float:
    decision_tree_model = joblib.load(
        'web/ML/saved_models/decision_tree_model.pkl')
    vectorizer = joblib.load(
        'web/ML/saved_models/decision_tree_vectorizer.pkl')

    def classify_article(title, text):
        combined = title + " " + text
        combined_vectorized = vectorizer.transform([combined])
        prediction = decision_tree_model.predict(combined_vectorized)
        pred_proba = decision_tree_model.predict_proba(combined_vectorized)

        proba_real = pred_proba[0][1] * 100
        return proba_real

    proba_real = classify_article(title, text)
    return float(f'{proba_real:.2f}')


def fetch_prediction_logistic_regression(title: str, text: str):
    logistic_regression_model = joblib.load(
        'web/ML/saved_models/logistic_regression_model.pkl')
    vectorizer = joblib.load(
        'web/ML/saved_models/logistic_regression_vectorizer.pkl')

    def classify_article(title, text):
        combined = title + " " + text
        processed_text = preprocess(combined)
        combined_vectorized = vectorizer.transform([processed_text])
        pred_proba = logistic_regression_model.predict_proba(
            combined_vectorized)

        proba_real = pred_proba[0][1] * 100
        proba_fake = pred_proba[0][0] * 100
        print(f"Logistic reg = Probabilities: Real: {
              proba_real:.2f}%, Fake: {proba_fake:.2f}%")
        return proba_real

    return float(f'{classify_article(title, text):.2f}')


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


def fetch_prediction_multinomial_nb(title: str, text: str):
    multinomial_nb_model = joblib.load(
        'web/ML/saved_models/multinomial_NB_model.pkl')
    vectorizer = joblib.load(
        'web/ML/saved_models/multinomial_NB_vectorizer.pkl')

    def classify_article(title: str, text: str):
        combined = title + " " + text
        processed_text = preprocess(combined)
        combined_vectorized = vectorizer.transform([processed_text])
        pred_proba = multinomial_nb_model.predict_proba(combined_vectorized)

        proba_real = pred_proba[0][1] * 100
        proba_fake = pred_proba[0][0] * 100
        print(f"Multilomial = Probabilities: Real: {
              proba_real:.2f}%, Fake: {proba_fake:.2f}%")
        return proba_real

    return float(f'{classify_article(title, text):.2f}')


def fetch_prediction_svm(title: str, text: str):
    svm_model = joblib.load('web/ML/saved_models/svm_model.pkl')
    vectorizer = joblib.load('web/ML/saved_models/svm_vectorizer.pkl')

    def classify_article(title, text):
        combined = title + " " + text
        processed_text = preprocess(combined)
        combined_vectorized = vectorizer.transform([processed_text])
        prediction = svm_model.predict(combined_vectorized)

        decision_function = svm_model.decision_function(combined_vectorized)
        proba = expit(decision_function)

        label = prediction[0]
        return label, proba

    label, proba = classify_article(title, text)

    proba_real = proba[0]
    proba_fake = 1 - proba[0]

    print(f"Probabilities: Real: {proba_real *
          100:.2f}%, Fake: {proba_fake * 100:.2f}%")
    return float(f'{proba_real * 100:.2f}')
