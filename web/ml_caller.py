import concurrent.futures
import random
from models import News


def predictFake(news: News) -> bool:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(func, news) for func in [
            __predict0, __predict1, __predict2, __predict3]]
        results = [f.result()
                   for f in concurrent.futures.as_completed(futures)]

    average = sum(results) / len(results)
    return average > 0.5


def __predict0(news: News) -> float:
    __cpu_bound_task(40)
    return random.random()


def __predict1(news: News) -> float:
    __cpu_bound_task(40)
    return random.random()


def __predict2(news: News) -> float:
    __cpu_bound_task(40)
    return random.random()


def __predict3(news: News) -> float:
    __cpu_bound_task(40)
    return random.random()


def __cpu_bound_task(n: int) -> int:
    return n if n <= 1 else __cpu_bound_task(n-1) + __cpu_bound_task(n-2)
