import asyncio
import base64
import time
from contextlib import asynccontextmanager
from io import BytesIO

import torch
from PIL import Image
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from torchvision.models import resnet18, ResNet18_Weights

from host.db import APIRequest, User, get_session

# Brug grafikkortet ("cuda" KUN Nvidia), ("mps" kun for Mac-users grafikkort), hvis der er et, ellers processoren ("cpu")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Bruger device: {device}")

ml = {}  # holds the model and its peripherals

@asynccontextmanager
async def lifespan(app: FastAPI):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(device)
    model.eval()
    ml["model"] = model
    ml["preprocess"] = weights.transforms()
    ml["categories"] = weights.meta["categories"]
    print("Model indlæst!")
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
        return {"reply": f"Du skrev: {req.message}"}
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
