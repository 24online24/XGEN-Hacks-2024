# import concurrent.futures
from urllib.parse import unquote
from ML.data_fetcher import fetch_prediction_decision_tree, fetch_prediction_logistic_regression, fetch_prediction_multinomial_nb, fetch_prediction_random_forest, fetch_prediction_svm
from models import News


def predict_real(news: News) -> dict[str, float]:
    # model_name_to_function = {
    #     "Random Forest": fetch_prediction_random_forest,
    #     "Decision Tree": fetch_prediction_decision_tree,
    #     "Logistic Regression": fetch_prediction_logistic_regression,
    #     "Multinomial Naive Bayes": fetch_prediction_multinomial_nb,
    #     "Support Vector Machine": fetch_prediction_svm
    # }

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     function_call_to_model_name = {executor.submit(function, deserialized_title, deserialized_content): model_name
    #                                    for model_name, function in model_name_to_function.items()}
    #     model_name_to_result = {function_call_to_model_name[function_call]: function_call.result()
    #                             for function_call in concurrent.futures.as_completed(function_call_to_model_name)}

    # model_name_to_result = {
    #     "Random Forest": fetch_prediction_random_forest(news.title, news.content),
    #     "Decision Tree": fetch_prediction_decision_tree(news.title, news.content),
    #     "Logistic Regression": fetch_prediction_logistic_regression(news.title, news.content),
    #     "Multinomial Naive Bayes": fetch_prediction_multinomial_nb(news.title, news.content),
    #     "Support Vector Machine": fetch_prediction_svm(news.title, news.content)
    # }

    deserialized_title = unquote(news.title)
    deserialized_content = unquote(news.content)

    return {
        "Random Forest": fetch_prediction_random_forest(deserialized_title, deserialized_content),
        "Decision Tree": fetch_prediction_decision_tree(deserialized_title, deserialized_content),
        "Logistic Regression": fetch_prediction_logistic_regression(deserialized_title, deserialized_content),
        "Multinomial Naive Bayes": fetch_prediction_multinomial_nb(deserialized_title, deserialized_content),
        "Support Vector Machine": fetch_prediction_svm(deserialized_title, deserialized_content)
    }
