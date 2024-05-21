from urllib.parse import unquote
from ML.data_fetcher import fetch_prediction_decision_tree, fetch_prediction_logistic_regression, fetch_prediction_multinomial_nb, fetch_prediction_random_forest, fetch_prediction_svm
from models import News


def predict_real(news: News) -> dict[str, float]:
    deserialized_title = unquote(news.title)
    deserialized_content = unquote(news.content)

    return {
        "Random Forest": fetch_prediction_random_forest(deserialized_title, deserialized_content),
        "Decision Tree": fetch_prediction_decision_tree(deserialized_title, deserialized_content),
        "Logistic Regression": fetch_prediction_logistic_regression(deserialized_title, deserialized_content),
        "Multinomial Naive Bayes": fetch_prediction_multinomial_nb(deserialized_title, deserialized_content),
        "Support Vector Machine": fetch_prediction_svm(deserialized_title, deserialized_content)
    }
