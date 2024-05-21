import concurrent.futures
from ML.data_fetcher import fetch_prediction_decision_tree, fetch_prediction_logistic_regression, fetch_prediction_multinomial_nb, fetch_prediction_random_forest, fetch_prediction_svm
from models import News


def predict_real(news: News) -> dict[str, float]:
    model_name_to_function = {
        "random_forest": fetch_prediction_random_forest,
        "decision_tree": fetch_prediction_decision_tree,
        "logistic_regression": fetch_prediction_logistic_regression,
        "multinomial_nb": fetch_prediction_multinomial_nb,
        "svm": fetch_prediction_svm
    }

    with concurrent.futures.ProcessPoolExecutor() as executor:
        function_call_to_model_name = {executor.submit(function, news.title, news.content): model_name
                                       for model_name, function in model_name_to_function.items()}
        model_name_to_result = {function_call_to_model_name[function_call]: function_call.result()
                                for function_call in concurrent.futures.as_completed(function_call_to_model_name)}

    return model_name_to_result
