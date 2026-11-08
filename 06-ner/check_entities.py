"""
เช็กชื่อเฉพาะ (NER) และตัวเลขในบทสรุปที่โมเดลสร้าง เทียบกับต้นข่าวและบทสรุปอ้างอิง

ไฟล์ input ต้องมีคอลัมน์: body (ต้นข่าว), summary (บทสรุปอ้างอิง), pred (บทสรุปของโมเดล)

ใช้งาน:
    pip install pythainlp python-crfsuite pandas
    python check_entities.py predictions_mt5.csv --model mt5
    python check_entities.py predictions_mbart.csv --model mbart

ผลลัพธ์:
    ner_check_<model>.csv      รายละเอียดทีละข่าว
    ner_summary_<model>.json   ค่าเฉลี่ยรวม (เอาไปรวมผลกับโมเดลอื่นได้)
"""
import argparse
import json
import re
from difflib import SequenceMatcher

import pandas as pd
from pythainlp.tag import NER
from pythainlp.util import normalize as th_normalize

ENTITY_TYPES = {"PERSON", "ORGANIZATION", "LOCATION"}
TITLES = ("นางสาว", "นาย", "นาง", "น.ส.", "ด.ช.", "ด.ญ.")
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

ner = NER("thainer")


def norm(text: str) -> str:
    """ตัดช่องว่างและคำนำหน้าชื่อ เพื่อเทียบข้อความแบบไม่สนรูปแบบ"""
    # รวม "ํ" + "า" ให้เป็น "ำ" และเรียงวรรณยุกต์ให้เป็นแบบเดียวกัน
    text = th_normalize(str(text)).replace("\u0e4d\u0e32", "\u0e33")
    text = re.sub(r"\s+", "", text)
    for t in TITLES:
        if text.startswith(t) and len(text) > len(t) + 1:
            text = text[len(t):]
            break
    return text


def extract_entities(text: str) -> set:
    """ดึงชื่อเฉพาะ (คน องค์กร สถานที่) จากผล BIO ของ ThaiNER"""
    if not isinstance(text, str) or not text.strip():
        return set()
    tags = ner.tag(text, pos=False)
    ents, cur, cur_type = [], "", None
    for word, tag in tags:
        if tag.startswith("B-"):
            if cur and cur_type in ENTITY_TYPES:
                ents.append(cur)
            cur, cur_type = word, tag[2:]
        elif tag.startswith("I-") and cur_type == tag[2:]:
            cur += word
        else:
            if cur and cur_type in ENTITY_TYPES:
                ents.append(cur)
            cur, cur_type = "", None
    if cur and cur_type in ENTITY_TYPES:
        ents.append(cur)
    return {e.strip() for e in ents if len(norm(e)) >= 2}


def extract_numbers(text: str) -> set:
    return {n.replace(",", "") for n in NUM_RE.findall(str(text))}


def copy_ratio(pred: str, body: str) -> float:
    """สัดส่วนของบทสรุปที่เป็นข้อความต่อเนื่องก้อนเดียวลอกมาจากต้นข่าว"""
    if not pred:
        return 0.0
    sm = SequenceMatcher(None, pred, body, autojunk=False)
    m = sm.find_longest_match(0, len(pred), 0, len(body))
    return m.size / len(pred)


def check_row(body: str, summary: str, pred: str) -> dict:
    body_n = norm(body)
    pred_n = norm(pred)

    pred_ents = extract_entities(pred)
    ref_ents = extract_entities(summary)

    # ชื่อในบทสรุปโมเดลที่ไม่พบในต้นข่าวเลย = น่าจะแต่งขึ้นเอง
    unsupported = sorted(e for e in pred_ents if norm(e) not in body_n)
    # ชื่อในบทสรุปอ้างอิงที่โมเดลไม่ได้พูดถึง
    missed = sorted(e for e in ref_ents if norm(e) not in pred_n)

    body_nums = extract_numbers(body)
    pred_nums = extract_numbers(pred)
    wrong_nums = sorted(n for n in pred_nums if n not in body_nums)

    return {
        "n_pred_ents": len(pred_ents),
        "n_unsupported_ents": len(unsupported),
        "unsupported_ents": " | ".join(unsupported),
        "n_ref_ents": len(ref_ents),
        "n_missed_ents": len(missed),
        "missed_ents": " | ".join(missed),
        "n_pred_nums": len(pred_nums),
        "n_unsupported_nums": len(wrong_nums),
        "unsupported_nums": " | ".join(wrong_nums),
        "pred_copy_ratio": round(copy_ratio(str(pred), str(body)), 3),
        "ref_copy_ratio": round(copy_ratio(str(summary), str(body)), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="csv ที่มีคอลัมน์ body, summary, pred")
    ap.add_argument("--model", default="model", help="ชื่อโมเดล ใช้ตั้งชื่อไฟล์ผลลัพธ์")
    ap.add_argument("--limit", type=int, default=None, help="รันแค่ N แถวแรก (ไว้ทดสอบ)")
    args = ap.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    missing = {"body", "summary", "pred"} - set(df.columns)
    if missing:
        raise SystemExit(f"ไม่พบคอลัมน์: {missing} (มีอยู่: {list(df.columns)})")
    if args.limit:
        df = df.head(args.limit)

    rows = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        rows.append(check_row(str(r.body), str(r.summary), str(r.pred)))
        if i % 50 == 0:
            print(f"  {i}/{len(df)}")
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out.to_csv(f"ner_check_{args.model}.csv", index=False, encoding="utf-8-sig")

    tot_pred = out["n_pred_ents"].sum()
    tot_ref = out["n_ref_ents"].sum()
    tot_nums = out["n_pred_nums"].sum()
    summary = {
        "model": args.model,
        "n_articles": len(out),
        # ยิ่งต่ำยิ่งดี
        "entity_hallucination_rate": round(out["n_unsupported_ents"].sum() / max(tot_pred, 1), 4),
        "articles_with_hallucinated_entity": int((out["n_unsupported_ents"] > 0).sum()),
        "number_hallucination_rate": round(out["n_unsupported_nums"].sum() / max(tot_nums, 1), 4),
        # ยิ่งสูงยิ่งดี
        "entity_recall_vs_reference": round(1 - out["n_missed_ents"].sum() / max(tot_ref, 1), 4),
        # ไว้ดูว่าโมเดล/บทสรุปอ้างอิงลอกต้นข่าวมาแค่ไหน
        "mean_pred_copy_ratio": round(out["pred_copy_ratio"].mean(), 4),
        "mean_ref_copy_ratio": round(out["ref_copy_ratio"].mean(), 4),
    }
    with open(f"ner_summary_{args.model}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
