import asyncio
import base64
import json
import time
from contextlib import asynccontextmanager
from io import BytesIO
from threading import Thread

import torch
from PIL import Image
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from db import APIRequest, User, engine, get_session

# Brug grafikkortet ("cuda" KUN Nvidia), ("mps" kun for Mac-users grafikkort), hvis der er et, ellers processoren ("cpu")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Bruger device: {device}")

ml = {}  # holds the model and its peripherals

# Lille sprogmodel fra HuggingFace, som serveren selv kører (ingen ekstern AI-udbyder)
LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 256

@asynccontextmanager
async def lifespan(app: FastAPI):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(device)
    model.eval()
    ml["model"] = model
    ml["preprocess"] = weights.transforms()
    ml["categories"] = weights.meta["categories"]

    ml["tokenizer"] = AutoTokenizer.from_pretrained(LLM_NAME)
    llm = AutoModelForCausalLM.from_pretrained(LLM_NAME).to(device)
    llm.eval()
    ml["llm"] = llm
    print("Modeller indlæst!")
    yield
    ml.clear()  # release on shutdown

app = FastAPI(title="My AI API Server", version="1.0.0", lifespan=lifespan)

class ImageRequest(BaseModel):
    image: str  # Base64-encoded image bytes

class ChatRequest(BaseModel):
    message: str


# Hjemmesiden: "Dynamic API Keys with a Database"
bearer_scheme = HTTPBearer()

def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_session),
) -> User:
    user = db.scalar(select(User).where(User.api_key == creds.credentials))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return user

# Hjemmesiden: "Loggin each Request"
@app.post("/v1/chat")
async def chat(
    req: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    start = time.perf_counter()
    status_code = 200
    try:
        reply = await asyncio.to_thread(run_llm, req.message)
        return {"model": LLM_NAME, "reply": reply}
    except Exception as e:
        status_code = 500
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.add(APIRequest(
            user_id=user.id,
            endpoint="/v1/chat",
            response_time_ms=(time.perf_counter() - start) * 1000,
            status_code=status_code,
        ))
        db.commit()


# Hjemmesiden: "Streaming Responses with SSE" - svaret sendes del for del
@app.post("/v1/chat/stream")
async def chat_stream(
    req: ChatRequest,
    user: User = Depends(get_current_user),
):
    user_id = user.id  # gem id nu, generatoren kører efter dependencies er lukket

    # Almindelig (ikke-async) generator: FastAPI kører den i en tråd,
    # så ventetiden på modellen blokerer ikke resten af serveren
    def token_stream():
        start = time.perf_counter()
        status_code = 200
        try:
            # Modellen genererer i sin egen tråd og lægger tekst i streameren, mens vi læser
            streamer = TextIteratorStreamer(ml["tokenizer"], skip_prompt=True, skip_special_tokens=True)
            Thread(target=run_llm, args=(req.message, streamer)).start()
            for text in streamer:
                if text:
                    yield f"data: {json.dumps({'delta': text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception:
            status_code = 500
            raise
        finally:
            # Egen session, da Depends(get_session) kan være lukket når streamen kører
            with Session(engine) as db:
                db.add(APIRequest(
                    user_id=user_id,
                    endpoint="/v1/chat/stream",
                    response_time_ms=(time.perf_counter() - start) * 1000,
                    status_code=status_code,
                ))
                db.commit()

    return StreamingResponse(token_stream(), media_type="text/event-stream")


# Hjemmesiden: "Logging Each Request"
@app.post("/v1/classify")
async def classify(
    req: ImageRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    start = time.perf_counter()
    status_code = 200
    try:
        encoded = req.image.split(",", 1)[1] if req.image.startswith("data:") else req.image
        image = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
        predictions = await asyncio.to_thread(run_inference, image)
        return {"model": "resnet18", "predictions": predictions}
    except HTTPException as e:
        status_code = e.status_code
        raise
    except Exception as e:
        status_code = 500
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.add(APIRequest(
            user_id=user.id,
            endpoint="/v1/classify",
            response_time_ms=(time.perf_counter() - start) * 1000,
            status_code=status_code,
        ))
        db.commit()

def run_llm(message: str, streamer: TextIteratorStreamer | None = None) -> str:
    """Kører sprogmodellen på brugerens besked. Med en streamer sendes teksten ud løbende."""
    tokenizer = ml["tokenizer"]
    messages = [
        {"role": "system", "content": "Du er en hjælpsom assistent. Svar kort og på samme sprog som brugeren."},
        {"role": "user", "content": message},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = ml["llm"].generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, streamer=streamer)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_inference(image: Image.Image):
    batch = ml["preprocess"](image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = ml["model"](batch).softmax(dim=1)[0]
    top5 = probs.topk(5)
    return [
        {"label": ml["categories"][idx], "confidence": round(score.item(), 4)}
        for score, idx in zip(top5.values, top5.indices)
    ]


# Hjemmesiden: "Logging Each Request"
@app.get("/v1/usage")
def usage(user: User = Depends(get_current_user), db: Session = Depends(get_session)):
    total = db.scalar(
        select(func.count()).select_from(APIRequest).where(APIRequest.user_id == user.id)
    )
    return {"user_id": user.id, "total_requests": total}
