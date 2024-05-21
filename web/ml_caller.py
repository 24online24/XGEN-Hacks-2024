from ML.data_fetcher import fetch_prediction_decision_tree, fetch_prediction_logistic_regression, fetch_prediction_multinomial_nb, fetch_prediction_random_forest, fetch_prediction_svm


def predict_real(title, content) -> dict[str, float]:
    return {
        "Random Forest": fetch_prediction_random_forest(title, content),
        "Decision Tree": fetch_prediction_decision_tree(title, content),
        "Logistic Regression": fetch_prediction_logistic_regression(title, content),
        "Multinomial Naive Bayes": fetch_prediction_multinomial_nb(title, content),
        "Support Vector Machine": fetch_prediction_svm(title, content)
    }
