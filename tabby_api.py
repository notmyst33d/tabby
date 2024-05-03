import json
from llama_cpp import Llama
from aiohttp.web import Application, RouteTableDef, Request, StreamResponse, json_response, run_app

routes = RouteTableDef()

@routes.post("/prompt")
async def post_prompt(request: Request):
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST",
    }

    try:
        data = await request.json()
    except:
        return json_response({"error": "Bad JSON"}, status=400, headers=headers)

    if not data.get("messages"):
        return json_response({"error": "No messages"}, status=400, headers=headers)

    response = StreamResponse(headers=headers)
    await response.prepare(request)

    stream = request.app["llm"].create_chat_completion(
        [{"role": "system", "content": "Your name is Tabby"}] + data["messages"],
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        stream=True,
    )

    for completion in stream:
        await response.write(json.dumps({
            "token": completion["choices"][0]["delta"].get("content", ""),
            "stop": completion["choices"][0]["finish_reason"] == "stop",
        }).encode("utf-8") + b"\n")

    await response.write_eof()

app = Application()
app["llm"] = Llama(
    model_path="models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    use_mmap=False,
    use_mlock=True,
    n_threads=4,
    n_ctx=512,
    flash_attn=True,
    verbose=True,
)
app.add_routes(routes)
run_app(app, port=10000)
