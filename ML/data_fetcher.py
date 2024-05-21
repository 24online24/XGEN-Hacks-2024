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
    Random_Forest_Classifier = joblib.load("ML/saved_models/random_forest_model.pkl")
    vectorizer = joblib.load('ML/saved_models/random_forest_vectorizer.pkl')

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

def fetch_prediction_decision_tree(title: str, text: str) -> float:
    decision_tree_model = joblib.load('ML/saved_models/decision_tree_model.pkl')
    vectorizer = joblib.load('ML/saved_models/decision_tree_vectorizer.pkl')

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
    logistic_regression_model = joblib.load('ML/saved_models/logistic_regression_model.pkl')
    vectorizer = joblib.load('ML/saved_models/logistic_regression_vectorizer.pkl')

    def classify_article(title, text):
        combined = title + " " + text
        processed_text = preprocess(combined)
        combined_vectorized = vectorizer.transform([processed_text])
        pred_proba = logistic_regression_model.predict_proba(combined_vectorized)

        proba_real = pred_proba[0][1] * 100
        proba_fake = pred_proba[0][0] * 100
        print(f"Logistic reg = Probabilities: Real: {proba_real:.2f}%, Fake: {proba_fake:.2f}%")
        return proba_real

    return float(f'{classify_article(title, text):.2f}')
    # return classify_article(title, text)

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
    multinomial_nb_model = joblib.load('ML/saved_models/multinomial_NB_model.pkl')
    vectorizer = joblib.load('ML/saved_models/multinomial_NB_vectorizer.pkl')

    def classify_article(title: str, text: str):
        combined = title + " " + text
        processed_text = preprocess(combined)
        combined_vectorized = vectorizer.transform([processed_text])
        pred_proba = multinomial_nb_model.predict_proba(combined_vectorized)

        proba_real = pred_proba[0][1] * 100
        proba_fake = pred_proba[0][0] * 100
        print(f"Multilomial = Probabilities: Real: {proba_real:.2f}%, Fake: {proba_fake:.2f}%")
        return proba_real
    
    return float(f'{classify_article(title, text):.2f}')

def fetch_prediction_svm(title: str, text: str):
    svm_model = joblib.load('ML/saved_models/svm_model.pkl')
    vectorizer = joblib.load('ML/saved_models/svm_vectorizer.pkl')

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

    print(f"Probabilities: Real: {proba_real * 100:.2f}%, Fake: {proba_fake * 100:.2f}%")
    return float(f'{proba_real * 100:.2f}')
    
a = fetch_prediction_logistic_regression(
    title = "5 BIG LIES THE LEFTY MEDIA TOLD THIS WEEK…Then The Truth On Trump, Comey And Russia",
    text = "We have very few favorites when it comes to reporting for obvious reasons: They lie all the time! Who hasn t read or heard something totally false about Trump and Russia? This past week was a banner week of lies from the main stream media so it s our pleasure to straighten things out for everyone We ve discovered a star news reporting organization that stands above most others in their reporting on the Trump/Russia investigation. CIRCA NEWS with Sara Carter and John Solomon are setting the record straight!Here s the skinny on the latest media lies and the truth via Circa:Aggressive news reporting can be a public service, like when courageous journalists exposed Richard Nixon s Watergate, the Catholic church s cover up of the sexual abuse and the U.S. intelligence failures that preceded 9-11.But breathless, half-baked reporting in times of tumult can also misserve the public, like when The Wall Street Journal retracted a false story that Bill Clinton had been seen in a compromising position with an intern in the White House or when NBC wrongly identified Richard Jewell as the Olympic Park bombing suspect.This past week, professional journalism offered us several new examples of breathless reporting during the brouhaha over Donald Trump, James Comey and Russia intelligence. At their least, some stories misled the public, and at their worst they outright misinformed.Here are some examples this week that should cause the media to search whether its current standards are doing enough to ensure the public gets the whole truth. You can review the facts and decide for yourself whether the media shamed itself.The Rosenstein  Quitting Episode The Washington Post reported Wednesday night that Deputy Attorney General Rod Rosenstein had  threatened to resign after the narrative emerging from the White House on Tuesday evening cast him as a prime mover of the decision to fire Comey. The story cited an unnamed source close to the White House. But it did not have any comment or confirmation from the man who was alleged to have made the threat.When Sinclair Broadcast Group s Michelle Macaluso finally caught up to Rosenstein, a funny thing happened. He debunked the story. No, I m not quitting,  he said.The reporter pressed on:  Did you threaten to quit? No,  Rosenstein said.The Post did not return a call for comment Friday on whether it stood by its story.The Comey resources requestThe New York Times and several other outlets reported Wednesday that Comey, just before he was fired, had asked the Justice Department s Rosenstein for more funding and personnel for the Russia intelligence probe. But when Comey s deputy got to Capitol Hill the next day, he denied there was any need for more resources. I believe we have the adequate resources to do it and I know that we have resourced that investigation adequately,  FBI Deputy Director Andrew McCabe told lawmakers.McCabe said the FBI, if it needed more resources, wouldn t even go to the Justice Department but instead to Congress.  We don t typically request resources for an individual case,  he explained.CNN s claim that Trump  is under investigation During the breaking story on Comey Tuesday night, respected CNN legal analyst Jeffrey Toobin declared the FBI director s firing was a  grotesque abuse of power  and that it was a  political act when the President is under investigation. Toobin is entitled to his opinion but he should have the right facts. Numerous sources confirm to Circa that Comey told Congress just last week that Trump is NOT a target of the Russia probe.CNN s connection of grand jury subpoenas to Comey s firingCNN went live with an exclusive the night Comey was fired, reporting that grand jury subpoenas were issued to associates of former Trump National Security Adviser Michael Flynn seeking business records in the Russia case.But with one slight turn of hand, CNN s legitimate scoop was crafted to suggest there had been some correlation to Comey s firing. CNN learned of the subpoenas hours before President Donald Trump fired FBI director James Comey,  the network reported.The multitude of media comparisons to WatergateCountless media outlets from Politico and The New York Times to Mother Jones have suggested the whole Russia scandal is akin to Watergate, right down to Comey s firing mirroring Nixon s efforts to axe the special Watergate prosecutor Archibald Cox.There s just one problem with that. Nixon was the target of the Watergate probe and he was accused of specific crimes.Via: Circa New/John Solomon"
)

print(a)