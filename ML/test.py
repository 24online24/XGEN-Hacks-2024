# import joblib
# import pandas as pd
# import re
# import string

# Random_Forest_Classifier = joblib.load('ML/saved_models/random_forest2.pkl')
# vectorizer = joblib.load('ML/saved_models/vectorizer2.pkl')

# def preprocess(text):
#     text = text.lower()
#     text = re.sub('\[.*?\]', '', text)
#     text = re.sub("\\W", " ", text)
#     text = re.sub('https://\S+|www\.\S+', '', text)
#     text = re.sub('<.*?>+', '', text)
#     text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub('\n', '', text)
#     text = re.sub('\w*\d\w*', '', text)
#     return text

# def output_label(pred):
#     if pred == 1:
#         return "Fake News"
#     elif pred == 0:
#         return "Real News"

# def Fake_news_predict(news):
#     testing_news = {"text":[news]}
#     new_df_test = pd.DataFrame(testing_news)
#     new_df_test["text"] = new_df_test["text"].apply(preprocess)
#     new_x_test = vectorizer.transform(new_df_test["text"])
#     pred_RFC = Random_Forest_Classifier.predict(new_x_test)
#     pred_proba_RFC = Random_Forest_Classifier.predict_proba(new_x_test)

#     label = output_label(pred_RFC[0])
#     proba_real = pred_proba_RFC[0][0] * 100
#     proba_fake = pred_proba_RFC[0][1] * 100

#     return print(f"\n\nThis is {label}\nProbabilities: Real: {proba_real:.2f}%, Fake: {proba_fake:.2f}%")

# text = "Today in Baia Mare, the XGEN 2024 Conference took place, with approximately 190 papers registered from more than 20 universities from the country and abroad. The PNL candidate for the presidency of the Maramures County Council, Gabriel Stetco, was present with young people and spoke to them about the importance of active involvement in the community and the opportunities that Maramures County can offer for their professional and personal development. 'At the Maramures County Council, we have already started working on projects that support young people because we want the youth of Maramures to find all development opportunities at home. From this desire, the business incubator for young entrepreneurs was created, which will operate at the County Library, a project realized with European funds. We will continue to support as many projects for young people as possible because together we can build the future of Maramures,' said Gabriel Stetco. In his speech, Gabriel Stetco also highlighted some of the projects initiated by the County Council that support young people and contribute to ensuring a prosperous future for future generations. 'The 7 smart specialization parks, the dual education campus, and the opening of an extension of the 'Iuliu Hatieganu' University of Medicine and Pharmacy Cluj-Napoca in Baia Mare are other major projects we have successfully initiated in the last three and a half years, precisely to create conditions and development opportunities for young people in Maramures,' stated Gabriel Stetco, the PNL candidate for the presidency of the Maramures County Council. XGEN is a concept dedicated to the next generation (neXt GENeration) around which a group of teachers, students, representatives of the economic environment, and county and local authorities work, aiming to offer the next generation chances for a better life in the communities they belong to, without having to move to other cities or countries for a better life and personal and professional development opportunities. 'Supporting performance in education by awarding students who achieve excellent results in Olympiads, competitions, and national exams is another way we support young people and encourage them to work hard because only through education and concrete programs for the new generation can we build a developed, modern, and truly European society,' added Gabriel Stetco."

# Fake_news_predict(text)

import joblib
import pandas as pd
import re
import string
from scipy.special import expit  # Sigmoid function

svm_model = joblib.load('ML/saved_models/svm_model.pkl')
vectorizer = joblib.load('ML/saved_models/vectorizer.pkl')


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


def output_label(pred):
    if pred == 1:
        return "Fake News"
    elif pred == 0:
        return "Real News"


def Fake_news_predict(title, text):
    combined = title + " " + text
    combined = preprocess(combined)
    combined_vectorized = vectorizer.transform([combined])
    pred_SVM = svm_model.predict(combined_vectorized)
    pred_proba_SVM = svm_model.decision_function(combined_vectorized)

    proba_real = expit(-pred_proba_SVM[0]) * 100
    proba_fake = expit(pred_proba_SVM[0]) * 100

    label = output_label(pred_SVM[0])

    return print(f"\n\nThis is {label}\nProbabilities: Real: {proba_real:.2f}%, Fake: {proba_fake:.2f}%")


title = "NASA's Perseverance Rover Discovers Organic Molecules on Mars"
text = """In a shocking turn of events, a leaked government document has revealed that the administration plans to implant tracking chips in all citizens by 2025. The document, which was obtained by an anonymous hacker, outlines a detailed strategy for implementing the program, including how the chips will be manufactured, distributed, and monitored.
According to the document, the tracking chips will be mandatory for all citizens, regardless of age or health condition. The government claims that the chips are necessary to improve national security and streamline healthcare services. However, many experts and civil rights groups are raising concerns about the potential for abuse and invasion of privacy.
The document also suggests that the chips will have the capability to monitor individuals' movements, record conversations, and even control certain bodily functions. This revelation has sparked outrage among the public, with many taking to social media to voice their opposition to the plan.
Despite the uproar, government officials have remained tight-lipped about the leaked document. When asked for a comment, a spokesperson for the administration neither confirmed nor denied the authenticity of the document but assured the public that their rights and freedoms are of utmost importance.
As the debate over the proposed tracking chips intensifies, citizens are left wondering what the future holds and whether their privacy will be protected in an increasingly surveilled society.
"""

Fake_news_predict(title, text)
