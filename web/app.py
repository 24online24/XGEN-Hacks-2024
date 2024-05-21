from dataclasses import dataclass
from robyn import Request, Response, Robyn, jsonify, serve_file, serve_html

app = Robyn(__file__)


@dataclass
class News:
    title: str
    content: str


@app.get("/")
async def index():
    return serve_html("./frontend/dist/index.html")


@app.get("/assets/:file")
async def assets(request: Request):
    file_name = request.path_params.get("file")
    return serve_file(f"./frontend/dist/assets/{file_name}")


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
    return Response(
        status_code=200,
        headers={"Content-Type": "application/json"},
        description=jsonify(news.__dict__),
    )


# @app.post("/")
# async def


if __name__ == "__main__":
    app.start(host="0.0.0.0", port=8080)
