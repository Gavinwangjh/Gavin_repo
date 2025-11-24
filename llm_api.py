# llm_api.py
import os
import requests

API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-1c18d8d7f9a54fad812457adc89a84f9")   # ⚠ 换成你的 key
API_URL = "https://api.deepseek.com/chat/completions"

def call_llm(messages, model="deepseek-chat", temperature=0.3):
    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
