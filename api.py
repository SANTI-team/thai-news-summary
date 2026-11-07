"""API + หน้าเว็บ "สรุปข่าววันนี้" (คนที่ 8 ต่อยอดจากไฟล์นี้)

รัน:   python api.py        แล้วเปิด http://127.0.0.1:8000  (เอกสาร API: /docs)
โมเดล: ตั้ง MODEL_DIR = โฟลเดอร์โมเดลที่คนที่ 4 เทรนแล้ว
       ไม่ตั้ง = ใช้ csebuetnlp/mT5_multilingual_XLSum (ยังไม่ได้เทรนกับ ThaiSum)
"""
import os
import re
import time
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

HERE = Path(__file__).parent
MODEL = os.environ.get("MODEL_DIR", "csebuetnlp/mT5_multilingual_XLSum")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"กำลังโหลดโมเดล {MODEL} บน {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(DEVICE).eval()

# ---------- ตัดประโยค (ใช้กับ Lead-3) ----------
try:
    from pythainlp.tokenize import sent_tokenize
    sent_tokenize("ทดสอบ ระบบ", engine="crfcut")
    def split_sentences(text):
        return [s.strip() for s in sent_tokenize(text, engine="crfcut") if s.strip()]
except Exception:
    # สำรอง: ข่าวไทยเว้นวรรคแทนจุดจบประโยค → แยกที่ช่องว่าง/ขึ้นบรรทัด แล้วรวมชิ้นที่สั้นเกินไป
    def split_sentences(text):
        out = []
        for piece in re.split(r"\s{1,}|\n", text.strip()):
            if out and len(out[-1]) < 40:
                out[-1] += " " + piece
            elif piece:
                out.append(piece)
        return out

# ---------- NER (คนที่ 6 ปรับปรุง) ----------
try:
    from pythainlp.tag import NER
    _ner = NER("thainer")
    _ner.tag("ทดสอบ")
except Exception:
    _ner = None


def entities(text):
    if _ner is None:
        return []
    found, cur, lab = [], "", None
    for word, tag in _ner.tag(text) + [("", "O")]:
        if tag.startswith("I-") and lab == tag[2:]:
            cur += word
            continue
        if cur and lab:
            found.append((cur, lab))
        cur, lab = (word, tag[2:]) if tag.startswith("B-") else ("", None)
    seen = []
    for t, l in found:
        if (t, l) not in seen:
            seen.append((t, l))
    return [{"text": t, "type": l} for t, l in seen]


# ---------- API ----------
api = FastAPI(title="Thai News Summarizer API", version="0.1")
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class NewsIn(BaseModel):
    text: str = Field(..., min_length=20, description="เนื้อข่าว")
    max_length: int = Field(96, ge=16, le=200, description="ความยาวบทสรุปสูงสุด (token)")


@api.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL, "device": DEVICE, "ner": _ner is not None}


@api.post("/api/summarize")
def summarize(item: NewsIn):
    text = item.text.strip()
    t0 = time.time()
    enc = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(**enc, max_new_tokens=item.max_length, num_beams=4, no_repeat_ngram_size=3)
    summary = tokenizer.decode(ids[0], skip_special_tokens=True)
    sents = split_sentences(text)
    return {
        "summary": summary,                               # Abstractive (mT5)
        "lead3": " ".join(sents[:3]),                     # Extractive baseline: 3 ประโยคแรก
        "entities": entities(text),                       # ชื่อเฉพาะในข่าว
        "stats": {
            "news_chars": len(text), "summary_chars": len(summary),
            "truncated": enc["input_ids"].shape[1] >= 512,  # ข่าวยาวเกิน 512 token ถูกตัดท้าย
            "seconds": round(time.time() - t0, 2), "model": MODEL,
        },
    }


api.mount("/", StaticFiles(directory=HERE / "web", html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="127.0.0.1", port=8000)
