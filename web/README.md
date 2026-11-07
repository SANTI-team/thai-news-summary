# คู่มือสำหรับคนทำเว็บแอป

ฝั่ง AI ทำเป็น API ไว้แล้ว เพื่อนทำแค่หน้าเว็บ แล้วส่งข้อความข่าวไปที่ API

## รัน API
```bash
pip install -r requirements.txt
python api.py
```
- หน้าเว็บตัวอย่าง: http://127.0.0.1:8000/ (ไฟล์ `web/index.html` แก้ได้เลย)
- เอกสาร API ลองยิงได้ในเบราว์เซอร์: http://127.0.0.1:8000/docs
- ใช้ WangchanBERTa: ตั้ง `MODEL_DIR=models/wangchanberta-fakenews` ก่อนรัน (ไม่ตั้ง = โมเดล TF-IDF ตัวอย่าง)

## `POST /api/predict`
ส่ง:
```json
{ "text": "รัฐบาลแจกเงินคนละ 3000 บาท กดลิงก์นี้ด่วน!!" }
```
ได้กลับ:
```json
{
  "label": "ปลอม",
  "confidence": 0.625,
  "probabilities": { "จริง": 0.1927, "บิดเบือน": 0.1823, "ปลอม": 0.625 },
  "warnings": ["ใช้เครื่องหมาย ! หลายตัว", "มีลิงก์ให้กด (ระวังมิจฉาชีพ)"],
  "entities": [ { "text": "รัฐบาล", "type": "ORGANIZATION" }, { "text": "บาท", "type": "MONEY" } ],
  "explanation": [ { "token": "แจก", "weight": 0.35 }, { "token": "เงิน", "weight": 0.37 } ]
}
```
| ช่อง | ความหมาย |
|---|---|
| `label` | ผลทำนาย: `จริง` / `ปลอม` / `บิดเบือน` |
| `confidence` | ความมั่นใจ 0–1 |
| `probabilities` | ความน่าจะเป็นทุกคลาส (ทำกราฟแท่งได้) |
| `warnings` | สัญญาณเตือน เช่น เร่งให้แชร์ กล่าวอ้างเกินจริง |
| `entities` | ชื่อเฉพาะในข่าว (คน องค์กร สถานที่ การแพทย์ เงิน) |
| `explanation` | น้ำหนักของแต่ละคำ (`null` = ไม่สำคัญ) ใช้ไฮไลต์คำได้ — ตอนใช้ WangchanBERTa ยังเป็น `null` ทั้งหมด รอคนที่ 7 |

`GET /api/health` → เช็กว่า API ทำงานและใช้โมเดลไหน

## เรียกจาก JavaScript
```js
const r = await fetch("http://127.0.0.1:8000/api/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text }),
}).then(res => res.json());
console.log(r.label, r.confidence);
```
เปิด CORS ไว้แล้ว หน้าเว็บที่รันคนละพอร์ต (React/Vite ฯลฯ) เรียกได้เลย

> ⚠️ ตอนนี้โมเดลเทรนจากข้อมูลตัวอย่าง ผลยังไม่แม่น ใช้ทดสอบหน้าเว็บได้ แต่ห้ามเอาไปรายงาน
