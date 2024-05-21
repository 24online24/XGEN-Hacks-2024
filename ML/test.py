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
    
a = fetch_prediction_random_forest(
    title = "Trump's son, close associates to appear before Senate",
    text = "WASHINGTON (Reuters) - President Donald Trump’s son Donald Trump Jr., son-in-law Jared Kushner and former campaign manager Paul Manafort have been asked to appear before U.S. Senate committees next week to answer questions about the campaign’s alleged connections to Russia, officials said on Wednesday. The three men are the closest associates of the president to be called to speak to lawmakers involved in probing Russian meddling in the 2016 U.S. presidential election and possible collusion with the Trump campaign. Trump, who came into office in January, has been dogged by allegations that his campaign officials were connected to Russia, which U.S. intelligence agencies have accused of interfering in last year’s election.  Trump has denied any collusion.  The U.S. Senate Judiciary Committee said on Wednesday that it had called Trump’s eldest son, Donald Trump Jr., and  Manafort to testify on July 26 at a hearing. The president’s son released emails earlier this month that showed him eagerly agreeing to meet last year with a woman he was told was a Russian government lawyer who might have damaging information about Democratic presidential candidate Hillary Clinton. The meeting was also attended by Manafort and Kushner, who is now a senior adviser at the White House. Kushner is scheduled to be interviewed by the Senate Intelligence Committee on Monday, July 24, behind closed doors. “Working with and being responsive to the schedules of the committees, we have arranged Mr. Kushner’s interview with the Senate for July 24,” Kushner’s attorney, Abbe Lowell, said in a statement. “He will continue to cooperate and appreciates the opportunity to assist in putting this matter to rest.” A special counsel, Robert Mueller, is also conducting an investigation of Russian meddling in the U.S. election and any collusion between Moscow and Trump’s campaign.  The issue has overshadowed Trump’s tenure in office and irritated the president, who told the New York Times on Wednesday that he would not have appointed ally and former Senator Jeff Sessions as attorney general if he had known Sessions would recuse himself from oversight of the Russia probe. “Sessions should have never recused himself, and if he was going to recuse himself, he should have told me before he took the job and I would have picked somebody else,” Trump said in the interview. Senator Sheldon Whitehouse, a Democratic member of the Judiciary Committee, said the committee’s hearing would enable the panel to begin to get testimony under oath. “There has been an enormous amount that has been said publicly but it’s not under oath, which means that people are free to omit matters or lie with relative impunity,” Whitehouse told CNN. The Senate Intelligence Committee is conducting one of the main investigations of Russia’s meddling in the 2016 U.S. election and possible collusion by Trump associates, but the Judiciary committee has been looking into related issues. The public Judiciary hearing on Wednesday will look into rules governing the registration of agents working for foreign governments in the United States and foreign attempts to influence U.S. elections. Chuck Grassley, the committee’s Republican chairman, has said he wanted to question the Trump associates, but has also raised concerns about why the Obama administration allowed Natalia Veselnitskaya, the Russian lawyer who attended the Trump Tower meeting in June, into the United States.  He also has called before the committee and threatened to subpoena Glenn Simpson, a co-founder of Fusion GPS, a firm that commissioned former British intelligence agent Christopher Steele to dig up opposition research on Trump, when he was a presidential candidate."
)

print(a)