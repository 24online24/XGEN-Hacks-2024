import time
from robyn import Request, Response, Robyn, jsonify, serve_html

from ml_caller import predictFake
from models import News


app = Robyn(__file__)

app.add_directory("/assets", "./web/frontend/dist/assets")


@app.get("/")
async def index():
    return serve_html("./web/frontend/dist/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/predict")
async def predict(request: Request) -> Response:
    title = request.query_params.get("title")
    if title is None:
        return Response(
            status_code=400,
            headers={"Content-Type": "application/json"},
            description=jsonify({"error": "title is required"}),
        )

    content = request.query_params.get("content")
    if content is None:
        return Response(
            status_code=400,
            headers={"Content-Type": "application/json"},
            description=jsonify({"error": "content is required"}),
        )

    news = News(title=title, content=content)
    start = time.time()
    prediction = predictFake(news)
    print(f"Time taken: {time.time() - start}")
    return Response(
        status_code=200,
        headers={"Content-Type": "application/json"},
        description=jsonify({"prediction": prediction}),
    )


if __name__ == "__main__":
    app.start(host="0.0.0.0", port=8080)
