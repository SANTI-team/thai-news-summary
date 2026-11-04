"""เดโมระบบตรวจจับข่าวปลอมภาษาไทย (Text Classification + NER + อธิบายผล)

รัน:  python app.py   แล้วเปิด http://127.0.0.1:7860

ตอนนี้ใช้ TF-IDF + Logistic Regression ที่เทรนจาก sample_data.csv (ข้อมูลตัวอย่าง)
พอคนที่ 4 fine-tune WangchanBERTa เสร็จ ให้ตั้งค่า MODEL_DIR ชี้ไปที่โฟลเดอร์โมเดล
แล้วระบบจะสลับไปใช้ Transformer ให้เอง:  set MODEL_DIR=models\\wangchanberta-fakenews
"""
import os
from pathlib import Path

import gradio as gr
import pandas as pd
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).parent
LABEL_COLORS = {"จริง": "#2e9e5b", "ปลอม": "#d64545", "บิดเบือน": "#e0a020"}


def tokenize(text):
    return [t for t in word_tokenize(text, engine="newmm") if t.strip()]


# ---------- 1) Text Classification ----------
class TfidfModel:
    name = "TF-IDF + Logistic Regression (baseline)"

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                   ngram_range=(1, 2), sublinear_tf=True)
        X = self.vec.fit_transform(df["text"])
        self.clf = LogisticRegression(max_iter=2000, C=5).fit(X, df["label"])

    def predict(self, text):
        probs = self.clf.predict_proba(self.vec.transform([text]))[0]
        return dict(zip(self.clf.classes_, map(float, probs)))

    def explain(self, text, label):
        """น้ำหนักของแต่ละคำต่อคลาสที่ทำนาย = ค่า TF-IDF x สัมประสิทธิ์ของโมเดล"""
        row = list(self.clf.classes_).index(label)
        vocab = self.vec.vocabulary_
        x = self.vec.transform([text])
        out = []
        for tok in tokenize(text):
            idx = vocab.get(tok)
            w = float(x[0, idx] * self.clf.coef_[row, idx]) if idx is not None else 0.0
            out.append((tok, w))
        top = max((abs(w) for _, w in out), default=0) or 1
        return [(t, round(w / top, 2) if abs(w) / top > 0.05 else None) for t, w in out]


class TransformerModel:
    def __init__(self, model_dir):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=model_dir, top_k=None)
        self.name = f"Transformer ({Path(model_dir).name})"

    def predict(self, text):
        return {r["label"]: float(r["score"]) for r in self.pipe(text, truncation=True)[0]}

    def explain(self, text, label):
        return [(t, None) for t in tokenize(text)]  # คนที่ 7 ใส่ LIME/SHAP ตรงนี้


model_dir = os.environ.get("MODEL_DIR")
model = TransformerModel(model_dir) if model_dir else TfidfModel(HERE / "sample_data.csv")


# ---------- 2) NER ----------
GAZETTEER = {  # ใช้แทนเมื่อโหลดโมเดล NER ไม่ได้
    "ORGANIZATION": ["กระทรวงสาธารณสุข", "กรมควบคุมโรค", "กรมอนามัย", "ธนาคารแห่งประเทศไทย",
                     "ธนาคาร", "รัฐบาล", "กรมอุตุนิยมวิทยา", "ตำรวจ", "โรงพยาบาล", "มหาวิทยาลัย"],
    "LOCATION": ["กรุงเทพ", "เชียงใหม่", "ภาคเหนือ", "ภาคใต้", "ภาคอีสาน", "ญี่ปุ่น", "ประเทศไทย"],
    "PERSON": ["หมอ", "นายก", "นักวิทยาศาสตร์", "ผู้เชี่ยวชาญ", "นักวิจัย"],
    "MEDICAL": ["มะเร็ง", "วัคซีน", "ยาฟ้าทะลายโจร", "โรคหัวใจ", "ไข้เลือดออก", "ไข้หวัดใหญ่", "ขมิ้นชัน"],
    "MONEY": ["บาท", "สตางค์", "ดอกเบี้ย", "ภาษี"],
}

try:
    from pythainlp.tag import NER
    _ner = NER("thainer")
    _ner.tag("ทดสอบ")
    NER_NAME = "PyThaiNLP ThaiNER"
except Exception:
    _ner = None
    NER_NAME = "พจนานุกรมคำ (fallback)"


def extract_entities(text):
    if _ner is not None:
        spans = []
        for word, tag in _ner.tag(text):
            label = tag.split("-", 1)[1] if "-" in tag else None
            if spans and label and tag.startswith("I-") and spans[-1][1] == label:
                spans[-1] = (spans[-1][0] + word, label)
            else:
                spans.append((word, label))
        return spans
    spans, i = [], 0
    terms = sorted(((t, lab) for lab, ts in GAZETTEER.items() for t in ts), key=lambda x: -len(x[0]))
    while i < len(text):
        hit = next(((t, lab) for t, lab in terms if text.startswith(t, i)), None)
        if hit:
            spans.append(hit)
            i += len(hit[0])
        else:
            if spans and spans[-1][1] is None:
                spans[-1] = (spans[-1][0] + text[i], None)
            else:
                spans.append((text[i], None))
            i += 1
    return spans


# ---------- 3) สัญญาณเตือนเชิงสไตล์ ----------
def style_flags(text):
    flags = []
    if text.count("!") >= 2:
        flags.append("ใช้เครื่องหมาย ! หลายตัว")
    for w in ["ด่วน", "แชร์", "ส่งต่อ", "บอกต่อ"]:
        if w in text:
            flags.append(f"เร่งให้แชร์/เร่งด่วน: “{w}”")
    for w in ["100%", "หายขาด", "ทุกโรค", "แน่นอน", "ไม่อยากให้รู้", "ปิดข่าว"]:
        if w in text:
            flags.append(f"กล่าวอ้างเกินจริง: “{w}”")
    if "ลิงก์" in text or "http" in text:
        flags.append("มีลิงก์ให้กด (ระวังมิจฉาชีพ)")
    return flags


def analyze(text):
    text = (text or "").strip()
    if not text:
        return {}, "", [], []
    probs = model.predict(text)
    label = max(probs, key=probs.get)
    flags = style_flags(text)
    color = LABEL_COLORS.get(label, "#666")
    verdict = (f"<div style='padding:14px;border-radius:10px;background:{color};color:white;"
               f"font-size:20px;font-weight:600'>ผลทำนาย: {label} ({probs[label]:.0%})</div>"
               + "<p style='margin-top:8px'><b>สัญญาณเตือน:</b> "
               + (" · ".join(flags) if flags else "ไม่พบ") + "</p>")
    return probs, verdict, extract_entities(text), model.explain(text, label)


EXAMPLES = [
    "ด่วน!! แชร์ต่อ ดื่มน้ำต้มใบเตยทุกวัน รักษามะเร็งหายขาด 100% หมอไม่อยากให้รู้",
    "กรมควบคุมโรคแนะนำให้ประชาชนในภาคเหนือระวังไข้เลือดออกในช่วงฤดูฝน",
    "ผลวิจัยพบว่าชาเขียวมีสารต้านอนุมูลอิสระ จึงสรุปว่าดื่มแทนยาเบาหวานได้เลย",
    "รัฐบาลแจกเงินคนละ 3000 บาท กดลิงก์นี้แล้วกรอกเลขบัตรประชาชนด่วน!!",
]

with gr.Blocks(title="ตรวจจับข่าวปลอมภาษาไทย") as demo:
    gr.Markdown("# ระบบตรวจจับข่าวปลอมและข้อมูลบิดเบือนภาษาไทย\n"
                f"โมเดลจำแนก: **{model.name}** · NER: **{NER_NAME}**  \n"
                "⚠️ เป็นระบบช่วยคัดกรองเบื้องต้น ไม่ใช่ผู้ตัดสินความจริง ควรตรวจสอบกับแหล่งข่าวทางการเสมอ")
    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Textbox(label="วางข้อความข่าว / โพสต์", lines=5)
            btn = gr.Button("วิเคราะห์", variant="primary")
            gr.Examples(EXAMPLES, inp)
        with gr.Column(scale=2):
            verdict = gr.HTML()
            probs = gr.Label(label="ความน่าจะเป็นแต่ละคลาส")
    ents = gr.HighlightedText(label="ชื่อเฉพาะที่ข่าวกล่าวถึง (NER)")
    expl = gr.HighlightedText(label="คำที่มีผลต่อการตัดสิน (สีเข้ม = ผลักไปทางผลทำนายมาก, สีตรงข้าม = ค้าน)")
    btn.click(analyze, inp, [probs, verdict, ents, expl])
    inp.submit(analyze, inp, [probs, verdict, ents, expl])

if __name__ == "__main__":
    demo.launch()
