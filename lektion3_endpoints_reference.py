#Kør server med kommando: fastapi dev "Lektion 3/main.py"

from fastapi import FastAPI, HTTPException
import time
import base64
import io
from PIL import Image
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
class ChatRequest(BaseModel):
    message: str

class ImageRequest(BaseModel):
    image: str

app = FastAPI(title="My API Server", version="1.0.0")

@app.get("/v1/hello")
def read_root():
    return {"message": "Welcome to my API server!"}

@app.post("/v1/chat")
def chat(data: ChatRequest):
    return {"reply": f"You said: {data.message}"}

def fake_token_stream():
    for word in ["Hello", "from", "my", "streaming", "API"]:
        yield f"data: {word}\n\n"
        time.sleep(0.3)
    yield "data: [DONE]\n\n"

@app.get("/v1/chat/stream")
def stream():
    return StreamingResponse(fake_token_stream(), media_type="text/event-stream")

@app.post("/v1/inspect")
def inspect(data: ImageRequest):
    try:
        encoded = data.image
        # Fjern evt. data-URL-prefix, fx "data:image/jpeg;base64,"
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        raw = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(raw))
        return {"format": img.format, "size": img.size, "mode": img.mode}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
