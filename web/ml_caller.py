import concurrent.futures
from models import News
# from ..ML.data_fetcher import fetch_prediction_random_forest


def predictReal(news: News) -> dict[str, float]:
    method_predictor_dic = {
        # "random_forest": fetch_prediction_random_forest,
        "predict0": __predict0,
        "predict1": __predict1,
        "predict2": __predict2,
        "predict3": __predict3,
        "predict4": __predict4,
    }

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(predictor, news.title, news.content): method_name
                   for method_name, predictor in method_predictor_dic.items()}
        results = {futures[f]: f.result()
                   for f in concurrent.futures.as_completed(futures)}

    return results


def __predict0(title: str, text: str) -> float:
    __cpu_bound_task(40)
    return 0.0


def __predict1(title: str, text: str) -> float:
    __cpu_bound_task(35)
    return 0.1


def __predict2(title: str, text: str) -> float:
    __cpu_bound_task(30)
    return 0.2


def __predict3(title: str, text: str) -> float:
    __cpu_bound_task(35)
    return 0.3


def __predict4(title: str, text: str) -> float:
    __cpu_bound_task(10)
    return 0.4


def __cpu_bound_task(n: int) -> int:
    return n if n <= 1 else __cpu_bound_task(n-1) + __cpu_bound_task(n-2)
