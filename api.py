# api.py
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"
app = FastAPI(title="EN->HI Translator")


class TranslateRequest(BaseModel):
    text: str


@app.on_event("startup")
def load():
    global tokenizer, model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model on", device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)


@app.post("/translate")
def translate(req: TranslateRequest):
    inputs = tokenizer(
        [req.text], return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_length=128, num_beams=4, early_stopping=True
        )
    translated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return {"en": req.text, "hi": translated}


@app.get("/")
def root():
    return {"status": "ready"}
