import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv("ML/csv_train/Combined.csv")

df['combined'] = df['title'] + " " + df['text']

texts = df['combined'].tolist()
labels = df['Label'].tolist()

labels = [int(label) for label in labels]

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(texts)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

joblib.dump(rf_model, 'ML/saved_models/random_forest_model.pkl')
joblib.dump(vectorizer, 'ML/saved_models/tfidf_vectorizer.pkl')