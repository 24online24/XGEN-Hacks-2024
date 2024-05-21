import time
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

df_fake = pd.read_csv("ML/csv_train/Fake.csv")
df_real = pd.read_csv("ML/csv_train/True.csv")

df_real['is_fake'] = 0
df_fake['is_fake'] = 1

df = pd.concat([df_real, df_fake])

df.drop(['subject', 'title', 'date'], axis=1, inplace=True)

random_indexes = np.random.randint(0, len(df), len(df))
df = df.iloc[random_indexes].reset_index(drop=True)


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


df['text'] = df['text'].apply(preprocess)

x = df["text"]
y = df["is_fake"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

vectorization = TfidfVectorizer()
x_train = vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test)

Random_Forest_Classifier = RandomForestClassifier(random_state=0)
Random_Forest_Classifier.fit(x_train, y_train)

joblib.dump(Random_Forest_Classifier,
            'web/ML/saved_models/random_forest_model.pkl')
joblib.dump(vectorization, 'web/ML/saved_models/random_forest_vectorizer.pkl')
