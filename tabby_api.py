import os
import json
from aiohttp.web import Application, RouteTableDef, Request, StreamResponse, json_response, run_app

backend = os.environ.get("BACKEND", "llamacpp")
hf_model = os.environ.get("HFMODEL")
hf_token = os.environ.get("HFTOKEN")
routes = RouteTableDef()

class HuggingfaceAdapter:
    def __init__(self, client):
        self.client = client

    def create_chat_completion(self, *args, **kwargs):
        if kwargs.get("top_k"):
            del kwargs["top_k"]
        kwargs["max_tokens"] = 12480
        kwargs["stream"] = True
        for completion in self.client.chat_completion(*args, **kwargs):
            if not completion.choices:
                yield {"error": "Context limit reached"}
            yield {"choices": [{"delta": {"content": completion.choices[0].delta.content}, "finish_reason": completion.choices[0].finish_reason}]}

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
        if response.get("error"):
            await response.write(json.dumps(response).encode("utf-8") + b"\n")
            break

        await response.write(json.dumps({
            "token": completion["choices"][0]["delta"].get("content", ""),
            "stop": completion["choices"][0]["finish_reason"] != None,
        }).encode("utf-8") + b"\n")

    await response.write_eof()

app = Application()
if backend == "huggingface":
    from huggingface_hub import InferenceClient
    client = InferenceClient(model=hf_model, token=hf_token)
    app["llm"] = HuggingfaceAdapter(client)
elif backend == "llamacpp":
    from llama_cpp import Llama
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
