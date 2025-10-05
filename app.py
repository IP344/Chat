from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# --- Load Model once on startup ---
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# --- Serve HTML UI ---
@app.get("/")
def home():
    return FileResponse("index.html")

# --- Chat endpoint ---
@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    prompt = data.get("prompt", "")

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8
    )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"reply": reply}