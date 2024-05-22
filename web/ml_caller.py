from constants import MODEL_NAME
from ML.data_fetcher import fetch_prediction_decision_tree, fetch_prediction_logistic_regression, fetch_prediction_multinomial_nb, fetch_prediction_random_forest, fetch_prediction_svm


def predict_real(title, content) -> dict[MODEL_NAME, float]:
    return {
        MODEL_NAME.RANDOM_FOREST: fetch_prediction_random_forest(title, content),
        MODEL_NAME.DECISION_TREE: fetch_prediction_decision_tree(title, content),
        MODEL_NAME.LOGISTIC_REGRESSION: fetch_prediction_logistic_regression(title, content),
        MODEL_NAME.MULTINOMIAL_NAIVE_BAYES: fetch_prediction_multinomial_nb(title, content),
        MODEL_NAME.SUPPORT_VECTOR_MACHINE: fetch_prediction_svm(title, content)
    }
