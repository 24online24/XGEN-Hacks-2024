# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
# import joblib, re, string

# df = pd.read_csv("ML/csv_train/Bigger.csv")

# def preprocess(text):
#     text = text.lower()
#     text = re.sub(r'\[.*?\]', '', text)
#     text = re.sub(r"\\W", " ", text)
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)
#     text = re.sub(r'<.*?>+', '', text)
#     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub(r'\n', '', text)
#     text = re.sub(r'\w*\d\w*', '', text)
#     return text

# df['text'] = df['text'].apply(preprocess)

# x = df["text"]
# y = df["is_fake"]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# vectorization = TfidfVectorizer()
# x_train = vectorization.fit_transform(x_train)
# x_test = vectorization.transform(x_test)

# Random_Forest_Classifier = RandomForestClassifier(random_state=0)
# Random_Forest_Classifier.fit(x_train, y_train)


# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
# import re, string

# df = pd.read_csv("ML/csv_train/Bigger.csv")

# def preprocess(text):
#     text = text.lower()
#     text = re.sub(r'\[.*?\]', '', text)
#     text = re.sub(r"\\W", " ", text)
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)
#     text = re.sub(r'<.*?>+', '', text)
#     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub(r'\n', '', text)
#     text = re.sub(r'\w*\d\w*', '', text)
#     return text

# df['text'] = df['text'].fillna('')

# df['text'] = df['text'].apply(preprocess)

# x = df["text"]
# y = df["label"]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# vectorization = TfidfVectorizer(stop_words='english')
# x_train = vectorization.fit_transform(x_train)
# x_test = vectorization.transform(x_test)

# Random_Forest_Classifier = RandomForestClassifier(random_state=0)
# Random_Forest_Classifier.fit(x_train, y_train)

# y_pred = Random_Forest_Classifier.predict(x_test)

# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print(f"Accuracy: {accuracy}")
# print(f"Classification Report:\n{report}")

# def predict_news(text):
#     processed_text = preprocess(text)
#     vectorized_text = vectorization.transform([processed_text])
#     prediction = Random_Forest_Classifier.predict(vectorized_text)
#     prediction_proba = Random_Forest_Classifier.predict_proba(vectorized_text)

#     label = "Fake News" if prediction[0] == 1 else "Real News"
#     proba_real = prediction_proba[0][0] * 100
#     proba_fake = prediction_proba[0][1] * 100

#     return f"{label}\nProbabilities: Real: {proba_real:.2f}%, Fake: {proba_fake:.2f}%"

# new_text = "Today in Baia Mare, the XGEN 2024 Conference took place, with approximately 190 papers registered from more than 20 universities from the country and abroad. The PNL candidate for the presidency of the Maramures County Council, Gabriel Stetco, was present with young people and spoke to them about the importance of active involvement in the community and the opportunities that Maramures County can offer for their professional and personal development."

# print(f"Prediction for the new text: {predict_news(new_text)}")