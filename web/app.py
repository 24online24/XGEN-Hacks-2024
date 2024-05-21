import json
import time
from urllib.parse import unquote
from robyn import ALLOW_CORS, Request, Response, Robyn, serve_html

from ml_caller import predict_real


app = Robyn(__file__)

ALLOW_CORS(app, origins=["http://localhost:5173"])

app.add_directory("/assets", "./web/frontend/dist/assets")


@app.get("/")
async def index():
    return serve_html("./web/frontend/dist/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/predict")
async def predict(request: Request) -> Response:
    start = time.time()
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

    deserialized_title = unquote(title)
    deserialized_content = unquote(content)
    model_name_to_result = predict_real(
        deserialized_title, deserialized_content)
    result_list: list[dict[str, str | float]] = []
    for model_name, result in model_name_to_result.items():
        result_list.append(
            {
                "name": model_name,
                "description": "This is the probability of the news being fake",
                "value": result
            }
        )
    print(f"Time taken: {time.time() - start}")
    return Response(
        status_code=200,
        headers={"Content-Type": "application/json"},
        description=json.dumps(result_list),
    )


if __name__ == "__main__":
    app.start(host="0.0.0.0", port=8080)
