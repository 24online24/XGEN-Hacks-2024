import joblib
import pandas as pd
import re
import string

def fetch_prediction_random_forest(title: str, text: str) -> float:
    Random_Forest_Classifier = joblib.load('ML/saved_models/random_forest2.pkl')
    vectorizer = joblib.load('ML/saved_models/vectorizer2.pkl')

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
        testing_news = {"text":[news]}
        new_df_test = pd.DataFrame(testing_news)
        new_df_test["text"] = new_df_test["text"].apply(preprocess)
        new_x_test = vectorizer.transform(new_df_test["text"])
        pred_proba_RFC = Random_Forest_Classifier.predict_proba(new_x_test)

        proba_real = pred_proba_RFC[0][0] * 100
        
        return float(f'{proba_real:.2f}')

    return Fake_news_predict(title + " " + text)
    