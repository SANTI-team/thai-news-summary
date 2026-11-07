"""API สำหรับเว็บแอป (คนที่ทำหน้าเว็บเรียกใช้ตรงนี้)

รัน:   python api.py         แล้วเปิดเอกสาร API ที่ http://127.0.0.1:8000/docs
ใช้โมเดลตัวเดียวกับ app.py (ตั้ง MODEL_DIR เพื่อใช้ WangchanBERTa ไม่ตั้ง = TF-IDF ตัวอย่าง)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import HERE, model, NER_NAME, extract_entities, style_flags

api = FastAPI(title="Thai Fake News API", version="0.1")
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class NewsIn(BaseModel):
    text: str = Field(..., min_length=1, examples=["ด่วน!! แชร์ต่อ ดื่มน้ำมะนาวรักษามะเร็งหายขาด"])


@api.get("/api/health")
def health():
    return {"status": "ok", "model": getattr(model, "name", "unknown"), "ner": NER_NAME}


@api.post("/api/predict")
def predict(item: NewsIn):
    text = item.text.strip()
    probs = model.predict(text)
    label = max(probs, key=probs.get)
    return {
        "label": label,                                   # "จริง" | "ปลอม" | "บิดเบือน"
        "confidence": round(probs[label], 4),
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "warnings": style_flags(text),                    # สัญญาณเตือนเชิงสไตล์
        "entities": [{"text": t, "type": lab} for t, lab in extract_entities(text) if lab],
        "explanation": [{"token": t, "weight": w} for t, w in model.explain(text, label)],
    }


# หน้าเว็บตัวอย่าง (เพื่อนแก้/แทนที่โฟลเดอร์ web/ ได้เลย)
api.mount("/", StaticFiles(directory=HERE / "web", html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="127.0.0.1", port=8000)
