# Start fra et lille Python-base-image
FROM python:3.13-slim

# Arbejdsmappe inde i containeren. Senere COPY og RUN sker her
WORKDIR /app

# Installer afhængigheder FØR koden kopieres ind,
# så pip install kun kører igen, når requirements.txt ændres
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NYT: Hent ResNet-18-vægtene nu, så de bliver gemt i imaget
RUN python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.DEFAULT)"

# NYT: Hent også sprogmodellen (LLM) fra HuggingFace, så den ikke skal downloades ved hver opstart
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"

# Kopier selve server-koden
COPY server.py db.py ./

# Dokumenterer hvilken port serveren lytter på
EXPOSE 8000

# Start FastAPI-appen med uvicorn på alle netværksinterfaces (0.0.0.0)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
