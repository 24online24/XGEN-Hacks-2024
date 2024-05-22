import json
from urllib.parse import unquote
from robyn import ALLOW_CORS, Request, Response, Robyn, serve_html

from ml_caller import predict_real


app = Robyn(__file__)

ALLOW_CORS(app, origins=["http://localhost:5173"])

app.add_directory("/assets", "./web/frontend/dist/assets")

MODEL_DESCRIPTION = {
    "Random Forest": "Random Forest is an ensemble learning method that builds multiple decision trees and outputs their majority vote for classification or average prediction for regression, enhancing accuracy and robustness.",
    "Decision Tree": "A Decision Tree is a model that splits data into branches based on feature values, leading to a prediction at each leaf node.",
    "Logistic Regression": "Logistic Regression is a statistical model that predicts the probability of a binary outcome using a linear combination of input features.",
    "Multinomial Naive Bayes": "Multinomial Naive Bayes is a probabilistic classifier that uses Bayes' theorem to predict categories based on the frequency of features in the input data.",
    "Support Vector Machine": "Support Vector Machine (SVM) is a supervised learning model that finds the optimal hyperplane to separate data points into distinct classes."
}

MODEL_ACCURACY = {
    "Random Forest": "97.57",
    "Decision Tree": "99.62",
    "Logistic Regression": "98.79",
    "Multinomial Naive Bayes": "93.83",
    "Support Vector Machine": "95.74"
}


@app.get("/")
async def index():
    return serve_html("./web/frontend/dist/index.html")


@app.get("/predict")
async def predict(request: Request) -> Response:
    title = request.query_params.get("title")
    if title is None:
        return Response(
            status_code=400,
            headers={"Content-Type": "application/json"},
            description=json.dumps({"error": "title is required"}),
        )

    content = request.query_params.get("content")
    if content is None:
        return Response(
            status_code=400,
            headers={"Content-Type": "application/json"},
            description=json.dumps({"error": "content is required"}),
        )

    model_name_to_result = predict_real(unquote(title), unquote(content))
    result_list: list[dict[str, str | float]] = []
    for model_name, result in model_name_to_result.items():
        result_list.append(
            {
                "name": model_name,
                "description": MODEL_DESCRIPTION[model_name],
                "accuracy": MODEL_ACCURACY[model_name],
                "value": result
            }
        )

    return Response(
        status_code=200,
        headers={"Content-Type": "application/json"},
        description=json.dumps(result_list),
    )


@app.post("/feedback")
async def feedback(request: Request) -> Response:
    data = request.json()
    print(data)
    return Response(
        status_code=200,
        headers={"Content-Type": "application/json"},
        description=json.dumps({"message": "Feedback received"}),
    )


if __name__ == "__main__":
    app.start(host="0.0.0.0", port=8080)
