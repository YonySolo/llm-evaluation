import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

MODELS = [
    "HuggingFaceTB/SmolLM3-3B:hf-inference",
    "mistralai/Mistral-7B-Instruct-v0.3:hf-inference",
    "google/gemma-2-2b-it:hf-inference",
]

def query_model(model, question):
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 100,
    }

    start = time.time()
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    end = time.time()

    latency = round(end - start, 2)

    if response.status_code == 200:
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
    else:
        answer = f"Error: {response.status_code}"

    return answer, latency

df = pd.read_csv("dataset.csv")

results = []

for model in MODELS:
    print(f"\nTesting: {model}")
    for _, row in df.iterrows():
        question = row["question"]
        expected = row["expected_answer"]
        category = row["category"]
        print(f"  Asking: {question[:50]}...")
        answer, latency = query_model(model, question)
        results.append({
            "model": model,
            "question": question,
            "expected_answer": expected,
            "model_answer": answer,
            "latency": latency,
            "category": category,
        })

results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)
print(f"\nDone! Results saved to results.csv")
