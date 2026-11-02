# client.py
# Klient der tester, at AI-API-serveren virker.
# Bygget på min klient fra Lektion 4 (celle 24 og 26), som igen bygger på Module A.2.
#
# Kør (stå i Exam-mappen, mens serveren kører):  python3 client.py
import base64
import os
from io import BytesIO

import requests
from PIL import Image

# Serverens adresse og API-nøgle kan ændres med miljøvariabler (Module A.2: "Storing Secrets")
BASE_URL = os.getenv("BASE_URL", "http://172.20.10.8:8000")
API_KEY = os.getenv("API_KEY", "sk-ven-002")


def encode_image(path):
    """Åbner et billede, gør det mindre og pakker det ind som base64-tekst."""
    image = Image.open(path).convert("RGB")  # PNG kan have gennemsigtighed, som JPEG ikke kan gemme
    image.thumbnail((1024, 1024))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def make_headers(api_key):
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def classify(image_text, api_key):
    """Sender et billede til /v1/classify og skriver svaret ud."""
    try:
        response = requests.post(
            f"{BASE_URL}/v1/classify",
            headers=make_headers(api_key),
            json={"image": image_text},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        print(f"  Statuskode: {response.status_code}  Model: {result['model']}")
        for gaet in result["predictions"]:
            print(f"    {gaet['label']:<25} {gaet['confidence']:.1%}")
    except requests.exceptions.Timeout:
        print("  The API took too long to respond.")
    except requests.exceptions.HTTPError as e:
        print(f"  The API returned an error: {e}")
        print(f"  Body: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"  Request failed: {e}")


def chat(question, api_key):
    """Sender et spørgsmål til /v1/chat og skriver svaret ud."""
    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat",
            headers=make_headers(api_key),
            json={"message": question},
            timeout=60,
        )
        response.raise_for_status()
        print(f"  Chat-svar: {response.json()['reply']}")
    except requests.exceptions.Timeout:
        print("  Chatten tog for lang tid at svare.")
    except requests.exceptions.HTTPError as e:
        print(f"  Chat-API'et returnerede en fejl: {e}")
        print(f"  Body: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"  Chat-kaldet fejlede: {e}")


def usage(api_key):
    """Henter antal kald fra /v1/usage."""
    try:
        response = requests.get(f"{BASE_URL}/v1/usage", headers=make_headers(api_key), timeout=10)
        response.raise_for_status()
        return response.json()["total_requests"]
    except requests.exceptions.RequestException as e:
        print(f"  Kunne ikke hente usage: {e}")
        return None


if __name__ == "__main__":
    print(f"Server: {BASE_URL}\n")

    before = usage(API_KEY)
    print(f"Kald før test: {before}\n")

    print("Test: Chat-spørgsmål")
    chat("Hvad er kunstig intelligens?", API_KEY)
    print()

    horse = encode_image("horse.png")
    cat = encode_image("cat.jpg")

    # Test 1-3: API-nøgler (Module A.4: uden nøgle, forkert nøgle, gyldig nøgle)
    for beskrivelse, key in [
        ("Uden nøgle (forventet: 401/403)", None),
        ("Forkert nøgle (forventet: 401)", "sk-forkert-999"),
        ("Gyldig nøgle, horse.png (forventet: 200)", API_KEY),
    ]:
        print(f"Test: {beskrivelse}")
        classify(horse, key)
        print()

    # Test 4: et andet billede
    print("Test: Gyldig nøgle, cat.jpg (forventet: 200)")
    classify(cat, API_KEY)
    print()

    # Test 5: ugyldigt billede
    # server.py fanger fejlen i sin generelle "except Exception" og svarer derfor 500
    print("Test: Ugyldigt billede (forventet: 500)")
    classify("ikke-et-billede", API_KEY)
    print()

    # Usage skal være steget med de kald, der havde en gyldig nøgle
    after = usage(API_KEY)
    print(f"Kald efter test: {after}")
    if before is not None and after is not None:
        print(f"Stigning: {after - before} (forventet: 4)")
