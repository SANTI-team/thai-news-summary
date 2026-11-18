# ระบบสรุปข่าวภาษาไทยรายวัน (SANTI-team)

โปรเจกต์วิชา NLP · 8 คน · ทำงานเป็นคู่ (4 คู่)

> **เปลี่ยนหัวข้อ** จากเดิม "ตรวจจับข่าวปลอม" เป็น "สรุปข่าว" เพราะมีชุดข้อมูลข่าวไทยพร้อมบทสรุปให้ใช้ได้เลย (ThaiSum)
> ไฟล์ `app.py`, `sample_data.csv`, `team-roles/` ยังเป็นของหัวข้อเดิม จะปรับตามมาทีหลัง

## โจทย์
รับข่าวภาษาไทย 1 ข่าว แล้วสรุปเป็นข้อความสั้น 1–3 ประโยค พร้อมบอกชื่อคน องค์กร และสถานที่สำคัญที่อยู่ในข่าว
จากนั้นเทียบสองวิธี
- **Extractive** หยิบประโยคจากข่าวเดิมมาใช้
- **Abstractive** ให้โมเดลเขียนบทสรุปใหม่ด้วยตัวเอง
## เว็บ
https://thai-news-summarizer.vercel.app

## สไลด์นำเสนอ
https://canva.link/vk41q574injj7f1

## ข้อมูล
[ThaiSum](https://huggingface.co/datasets/pythainlp/thaisum) ข่าวไทยราว 358,868 ข่าว (train) + 11,000 (val) + 11,000 (test) จากไทยรัฐ ไทยพีบีเอส ประชาไท และ The Standard แต่ละข่าวมีบทสรุปที่นักข่าวเขียนไว้ · สัญญาอนุญาต MIT
คอลัมน์ที่ใช้: `body` (เนื้อข่าว), `summary` (บทสรุป)

## ใครทำอะไร

| คู่ | คน | งาน | ไฟล์ notebook |
|---|---|---|---|
| **A · ข้อมูล** | 1 | เตรียม ThaiSum: ทำความสะอาด สุ่มชุดย่อย แบ่ง train/val/test แจกทุกคน + สถิติความยาว | `notebooks/01-data.ipynb` |
| | 6 | NER: ดึงคน/องค์กร/สถานที่ + เช็กว่าบทสรุปตกหล่นชื่อสำคัญไหม | `notebooks/06-ner.ipynb` |
| **B · Extractive** | 2 | TextRank / TF-IDF เลือกประโยคสำคัญ | `notebooks/02-textrank.ipynb` |
| | 3 | Lead-3 (3 ประโยคแรก) + เทียบกับคนที่ 2 | `notebooks/03-lead3.ipynb` |
| **C · Abstractive** | 4 | fine-tune **mT5** สรุปข่าว (โมเดลหลัก) | `notebooks/04-mt5.ipynb` |
| | 5 | ใช้ notebook ของคนที่ 4 เปลี่ยนเป็น **mBART-50** | `notebooks/05-mbart.ipynb` |
| **D · ผลและเว็บ** | 7 | วัด ROUGE รวมทุกโมเดล + วิเคราะห์บทสรุปที่ผิด | `notebooks/07-eval.ipynb`, `reports/` |
| | 8 | เว็บแอป "สรุปข่าววันนี้" (ใช้ `api.py`) | `web/` |

อ้างอิงของแต่ละเทคนิค: [`docs/references.md`](docs/references.md)

## กติการ่วม
- **ทุกคนใช้ train/val/test ชุดเดียวกันที่คนที่ 1 แจก** ห้ามแบ่งเอง ไม่งั้นเทียบผลกันไม่ได้
- **ตัววัดหลัก:** ROUGE-1 / ROUGE-2 / ROUGE-L (F1) · ต้องตัดคำไทยด้วย PyThaiNLP (`newmm`) ก่อนคิด ROUGE เพราะภาษาไทยไม่มีช่องว่างระหว่างคำ
- ทดสอบบนชุด test ชุดเดียวกัน (เช่น 1,000 ข่าวแรกของ test) ทุกโมเดล
- ข้อมูลและไฟล์โมเดลไม่ commit (ใหญ่เกินไป) แชร์ผ่าน Google Drive ของกลุ่ม

## การทำงานกับ Git
1. สร้าง branch ของตัวเอง เช่น `part-4-mt5`
2. commit เฉพาะไฟล์ของตัวเอง
3. เปิด Pull Request เข้า `main` ให้เพื่อนในคู่ช่วยดู

ไม่ถนัด Git: เข้าหน้า repo → `Add file → Upload files` → เลือก "Create a new branch… and start a pull request" → `Propose changes`
