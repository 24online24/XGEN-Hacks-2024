import json
from urllib.parse import unquote
from robyn import ALLOW_CORS, Request, Response, Robyn, serve_html

from constants import MODEL_ACCURACY, MODEL_DESCRIPTION
from ml_caller import predict_real


app = Robyn(__file__)

app.add_directory("/assets", "./web/frontend/dist/assets")


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
                "name": model_name.value,
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
