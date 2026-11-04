# ระบบตรวจจับข่าวปลอมภาษาไทย (SANTI-team)

โปรเจกต์วิชา NLP · 7 คน · ทำงานเป็นคู่

## ใครทำอะไร

| คน | งาน | ไฟล์ notebook |
|---|---|---|
| 1 | ผู้ดูแลข้อมูล (ทำความสะอาด, แบ่ง train/test) | `notebooks/01-data.ipynb` |
| 2 | TF-IDF + Logistic Regression | `notebooks/02-tfidf.ipynb` |
| 3 | เทียบ Naive Bayes / SVM | `notebooks/03-classic-compare.ipynb` |
| 4 | fine-tune WangchanBERTa | `notebooks/04-wangchanberta.ipynb` |
| 5 | เทียบ XLM-RoBERTa | `notebooks/05-xlmr.ipynb` |
| 6 | NER | `notebooks/06-ner.ipynb` |
| 7 | รวมผล + วิเคราะห์ข้อผิดพลาด + เดโม | `app.py`, `reports/` |

ดูภาพรวมได้ที่ `team-roles/team-roles.png`

## กติการ่วม

- **รูปแบบข้อมูล:** CSV คอลัมน์ `text,label` โดย label เป็น `ปลอม` หรือ `จริง`
- **ทุกคนใช้ train/test ชุดเดียวกันที่คนที่ 1 แจก** ห้ามแบ่งเอง ไม่งั้นเทียบผลกันไม่ได้
- **ตัววัดหลัก:** F1 (macro)
- `sample_data.csv` เป็นข้อความที่แต่งขึ้น 36 แถว ใช้ทดสอบว่าโค้ดรันได้เท่านั้น **ห้ามเอาผลไปใส่รายงาน**
- ข้อมูลจริงและไฟล์โมเดลไม่ commit (ใหญ่เกินไป) แชร์ผ่าน Google Drive ของกลุ่ม

## การทำงานกับ Git

1. สร้าง branch ของตัวเอง เช่น `part-4-wangchanberta`
2. commit เฉพาะไฟล์ของตัวเอง
3. เปิด Pull Request เข้า `main` ให้เพื่อนในคู่ช่วยดู

## รันเดโม

```bash
pip install -r requirements.txt
python app.py
```

ตอนนี้เดโมใช้ TF-IDF จาก `sample_data.csv` ถ้าคนที่ 4 เทรนเสร็จแล้ว ให้ตั้ง `MODEL_DIR=models/wangchanberta-fakenews` เดโมจะเปลี่ยนไปใช้ WangchanBERTa
