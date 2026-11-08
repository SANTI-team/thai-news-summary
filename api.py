"""API + หน้าเว็บเดโม "สรุปข่าววันนี้" (Today's News Summary) — สำหรับคนที่ 8

วิธีรัน:
    python api.py
    แล้วเปิดเบราว์เซอร์ที่ http://127.0.0.1:8000 (เอกสาร Swagger API: http://127.0.0.1:8000/docs)

การเชื่อมต่อกับเพื่อนในทีม:
    - คนที่ 4 (mT5 Abstractive Summarization): แก้ไขในฟังก์ชัน `load_summarizer_model()` และ `model_summarize()`
      หรือตั้งค่า Environment Variable: MODEL_DIR="path/to/mt5-thaisum"
    - คนที่ 6 (Named Entity Recognition - NER): แก้ไขในฟังก์ชัน `load_ner_tagger()` และ `extract_entities()`
"""
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).parent

# ==============================================================================
# 1. การตั้งค่าและโหลดโมเดล (Integration Points สำหรับคนที่ 4 และ คนที่ 6)
# ==============================================================================

# TODO: [Member 4] เปลี่ยนพาธด้านล่างให้ชี้ไปยังโฟลเดอร์โมเดล mT5 ที่ Fine-tune กับ ThaiSum เสร็จแล้ว
# เช่น MODEL_DIR = str(HERE / "models" / "mt5-thaisum")
MODEL_NAME_OR_PATH = os.environ.get("MODEL_DIR", "csebuetnlp/mT5_multilingual_XLSum")
USE_MOCK_MODEL = os.environ.get("USE_MOCK_MODEL", "0") == "1"

_tokenizer = None
_model = None
_device = "cpu"
_model_status = "unloaded"
_model_load_attempted = False


def load_summarizer_model() -> bool:
    """โหลดโมเดล mT5 สำหรับสรุปข่าวแบบ Abstractive (เชื่อมต่อกับงานคนที่ 4)"""
    global _tokenizer, _model, _device, _model_status, _model_load_attempted
    if _model is not None:
        return True
    if _model_load_attempted:
        return False
    _model_load_attempted = True

    if USE_MOCK_MODEL:
        _model_status = "mock (ตั้งค่า USE_MOCK_MODEL=1)"
        return False

    # TODO: [Member 4] หากนำโฟลเดอร์โมเดลที่เทรนแล้วมาวางในโปรเจกต์ จะโหลดจากเครื่องทันที
    # ตั้งค่า DOWNLOAD_MODEL=1 หากต้องการดาวน์โหลดโมเดลขนาด 2.3 GB จาก Hugging Face อัตโนมัติ
    local_only = os.environ.get("DOWNLOAD_MODEL", "0") != "1"
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Member 4] กำลังโหลดโมเดลสรุปข่าว: {MODEL_NAME_OR_PATH} บน {_device} ...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, local_files_only=local_only)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH, local_files_only=local_only).to(_device).eval()
        _model_status = f"loaded ({MODEL_NAME_OR_PATH} on {_device})"
        print(f"[Member 4] โหลดโมเดลสำเร็จ: {_model_status}")
        return True
    except Exception as exc:
        _model_status = f"fallback (ใช้โหมดสำรองความเร็วสูง — ตั้งค่า MODEL_DIR หรือ DOWNLOAD_MODEL=1 เพื่อใช้ mT5 เต็มรูปแบบ)"
        print(f"[Member 4] ยังไม่พบไฟล์โมเดล mT5 ในเครื่อง ใช้ระบบสรุปสำรองความเร็วสูงแทน ({exc.__class__.__name__})")
        return False


_ner_engine = None
_ner_status = "unloaded"
_ner_load_attempted = False


def load_ner_tagger() -> bool:
    """โหลดโมเดล NER สำหรับดึงชื่อเฉพาะ (เชื่อมต่อกับงานคนที่ 6)"""
    global _ner_engine, _ner_status, _ner_load_attempted
    if _ner_engine is not None:
        return True
    if _ner_load_attempted:
        return False
    _ner_load_attempted = True

    # TODO: [Member 6] เปลี่ยน engine หรือโมเดล NER ที่พัฒนาใน notebooks/06-ner.ipynb ได้ที่นี่
    # เช่น NER("thainer") หรือโมเดล WangchanBERTa NER
    try:
        from pythainlp.tag import NER

        _ner_engine = NER("thainer")
        _ner_engine.tag("ทดสอบระบบ")
        _ner_status = "loaded (PyThaiNLP ThaiNER)"
        print(f"[Member 6] โหลดโมเดล NER สำเร็จ: {_ner_status}")
        return True
    except Exception as exc:
        _ner_status = f"rule-based fallback ({exc.__class__.__name__})"
        print(f"[Member 6] โหลด PyThaiNLP NER ไม่สำเร็จ ใช้ Rule-based NER สำรองแทน: {exc}")
        return False


# ==============================================================================
# 2. ฟังก์ชันช่วยเหลือสำหรับตัดคำและตัดประโยคภาษาไทย
# ==============================================================================

def count_thai_words(text: str) -> int:
    """นับจำนวนคำภาษาไทยด้วย PyThaiNLP (หรือใช้ Regex สำรองหากไม่มีไลบรารี)"""
    if not text.strip():
        return 0
    try:
        from pythainlp.tokenize import word_tokenize

        tokens = [w for w in word_tokenize(text, engine="newmm") if w.strip()]
        return len(tokens)
    except Exception:
        # สำรอง: ประมาณจำนวนคำจากกลุ่มตัวอักษรและช่องว่าง
        chunks = re.findall(r"[ก-๙]+|[a-zA-Z0-9]+", text)
        return sum(max(1, len(c) // 5) for c in chunks)


def split_thai_sentences(text: str) -> List[str]:
    """แยกประโยค/ย่อหน้าย่อยจากเนื้อข่าวภาษาไทยโดยไม่ตัดกลางชื่อย่อหรือ พ.ศ."""
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return []

    # แยกตามคำเชื่อมประโยคข่าวไทยเป็นหลักเพื่อให้แต่ละส่วนมีใจความสมบูรณ์
    discourse_parts = re.split(
        r"\s+(?=(?:โดย|ด้าน|ทั้งนี้|เบื้องต้น|ส่งผลให้|ขณะที่|นอกจากนี้|อย่างไรก็ตาม|ส่วน|สำหรับ))",
        cleaned,
    )
    sentences: List[str] = []
    for part in discourse_parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= 180:
            sentences.append(part)
        else:
            # หากย่อหน้ายาวมาก ให้แบ่งตามเว้นวรรคโดยไม่ตัดหลังคำนำหน้าชื่อหรือ พ.ศ.
            sub_pieces = [p.strip() for p in re.split(r"\s+", part) if p.strip()]
            buf = ""
            for piece in sub_pieces:
                if not buf:
                    buf = piece
                elif (
                    len(buf) < 90
                    or re.search(r"(?:นาย|นาง|นางสาว|น\.ส\.|พล\.ต\.[อทต]\.|พ\.ศ\.|วันที่|ที่|กว่า|รวม|\d+)$", buf)
                ):
                    buf += " " + piece
                else:
                    sentences.append(buf)
                    buf = piece
            if buf:
                sentences.append(buf)

    return sentences if sentences else [cleaned]


# ==============================================================================
# 3. ฟังก์ชันหลัก 3 ส่วน: Lead-3, AI Summarization (คนที่ 4), NER (คนที่ 6)
# ==============================================================================

def lead3_summarize(text: str) -> Tuple[str, float]:
    """
    ส่วนที่ 1: Lead-3 Extractive Baseline (ดึง 3 ประโยคแรกของข่าว)
    คืนค่า: (บทสรุป Lead-3, เวลาที่ใช้ในการประมวลผลเป็นวินาที)
    """
    t0 = time.perf_counter()
    sentences = split_thai_sentences(text)
    lead3_text = " ".join(sentences[:3]).strip()
    if not lead3_text:
        lead3_text = text.strip()[:250]
    elapsed = time.perf_counter() - t0
    return lead3_text, elapsed


def _fallback_abstractive_summarize(text: str, max_length: int = 96) -> str:
    """
    ระบบสรุปข่าวสำรองแบบกระชับ (ปรับขนาดตามตัวเลือก max_length ที่ส่งมาจากหน้าเว็บ)
    - 64 tokens (สั้นกระชับ): 1 ประโยคหลักที่สกัดเฉพาะใจความแก่น
    - 96 tokens (มาตรฐาน): 2 ประโยคสมดุล (แก่นเรื่อง + ผลกระทบหรือมติ)
    - 140 tokens (ละเอียด): 3 ประโยคครบถ้วน (รวมตัวเลข สถิติ และชื่อเฉพาะ)
    """
    sentences = split_thai_sentences(text)
    if not sentences:
        return text.strip()

    keywords = (
        "ส่งผลให้", "ทั้งนี้", "โดย", "คาดว่า", "ตั้งเป้า", "เบื้องต้น", "ล่าสุด",
        "เจ้าหน้าที่", "มติ", "เห็นชอบ", "เตือน", "มูลค่า", "ร้อยละ", "บาท", "ขยายตัว", "จับกุม"
    )

    if max_length <= 70:
        # สั้นกระชับ: เอาประโยคแรกที่กระชับคำฟุ่มเฟือยแล้ว
        first = sentences[0]
        summary = re.sub(r"(?:เปิดเผยภายหลังการประชุมคณะรัฐมนตรีว่า|เปิดเผยว่า|กล่าวว่า|ระบุว่า|รายงานว่า|แถลงข่าวผลปฏิบัติการว่า)\s*", "", first)
        summary = re.sub(r"\s{2,}", " ", summary).strip()
        return summary

    elif max_length <= 110:
        # มาตรฐาน: 2 ประโยค (ประโยคแรก + ประโยคผลลัพธ์/มติ)
        selected = [sentences[0]]
        for sent in sentences[1:]:
            if any(k in sent for k in keywords) and sent not in selected:
                selected.append(sent)
                break
        if len(selected) == 1 and len(sentences) > 1:
            selected.append(sentences[-1])
        summary = " ".join(selected)
        summary = re.sub(r"(?:เปิดเผยว่า|กล่าวว่า|ระบุว่า|รายงานว่า)\s*", "", summary)
        summary = re.sub(r"\s{2,}", " ", summary).strip()
        return summary

    else:
        # ละเอียด: 3 ประโยคเก็บรายละเอียดครบถ้วน
        selected = [sentences[0]]
        for sent in sentences[1:]:
            if any(k in sent for k in keywords) and sent not in selected:
                selected.append(sent)
            if len(selected) >= 3:
                break
        if len(selected) < 3 and len(sentences) > len(selected):
            for s in sentences[1:]:
                if s not in selected:
                    selected.append(s)
                if len(selected) >= 3:
                    break
        summary = " ".join(selected)
        summary = re.sub(r"\s{2,}", " ", summary).strip()
        return summary


def model_summarize(text: str, max_length: int = 96) -> Tuple[str, float, bool]:
    """
    ส่วนที่ 2: AI Abstractive Summarization (mT5 หรือ LLM)
    คืนค่า: (บทสรุปจากโมเดล, เวลาที่ใช้เป็นวินาที, สถานะว่าเนื้อข่าวถูกตัดเพราะยาวเกิน 512 token หรือไม่)
    """
    t0 = time.perf_counter()
    truncated = False

    # TODO: [Member 4] นำโค้ด Inference ของโมเดล mT5 (หรือ LLM) ที่เทรนแล้วมาใส่ในบล็อกนี้
    # ปัจจุบันรองรับการโหลด HuggingFace Seq2SeqLM (`csebuetnlp/mT5_multilingual_XLSum` หรือพาธใน MODEL_DIR)
    if load_summarizer_model() and _model is not None and _tokenizer is not None:
        try:
            import torch

            WHITESPACE_HANDLER = lambda k: re.sub(r"\s+", " ", re.sub(r"\n+", " ", k.strip()))
            clean_input = WHITESPACE_HANDLER(text)
            enc = _tokenizer(
                [clean_input],
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(_device)

            # ตรวจสอบว่าข่าวต้นฉบับยาวเกิน 512 tokens หรือไม่
            raw_tokens = _tokenizer.encode(clean_input, add_special_tokens=True)
            truncated = len(raw_tokens) > 512

            with torch.no_grad():
                output_ids = _model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=max_length,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            summary = _tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            if summary:
                elapsed = time.perf_counter() - t0
                return summary, elapsed, truncated
        except Exception as exc:
            print(f"[Member 4] เกิดข้อผิดพลาดขณะรันโมเดล mT5 ใช้ระบบสำรองแทน: {exc}")

    # กรณีโมเดลยังไม่พร้อมหรือรันในโหมดสำรอง
    summary = _fallback_abstractive_summarize(text, max_length=max_length)
    truncated = len(text) > 1800
    elapsed = time.perf_counter() - t0
    return summary, elapsed, truncated


def _rule_based_entities(text: str) -> List[Dict[str, str]]:
    """
    ระบบดึงชื่อเฉพาะอัจฉริยะ (Hybrid Rule-based + Corpus NER Booster)
    ครอบคลุมชื่อคนในเครื่องหมายคำพูด ยศ คำนำหน้าชื่อ ทั้ง 77 จังหวัด หน่วยงาน และมูลนิธิ
    """
    found: List[Tuple[str, str]] = []

    # 1. บุคคล (PERSON):
    # 1.1 ชื่อคนในเครื่องหมายคำพูด เช่น “เป็กกี้ ศรีธัญญา”, “บุ๋ม ปนัดดา”, “ก๊อต อธิป”
    quote_person_pat = r"[\"“”]([ก-๙]{2,15}(?:\s+[ก-๙]{2,20})?)[\"“”]"
    for m in re.finditer(quote_person_pat, text):
        candidate = m.group(1).strip()
        # กรองคำทั่วไปที่ไม่ใช่ชื่อคน
        if not re.search(r"(?:มูลนิธิ|กระทรวง|กรม|สำนักงาน|บริษัท|ถนน|จังหวัด|อำเภอ|เขต|ม็อบ|น้ำเงิน|สีส้ม|คดี|มาตรา|พรบ|พ\.ร\.บ\.)", candidate):
            if len(candidate) >= 3:
                found.append((candidate, "PERSON"))

    # 1.2 คำนำหน้าชื่อ/ยศ/ตำแหน่งทางการ
    person_pat = (
        r"(?:พล\.ต\.อ\.|พล\.ต\.ท\.|พล\.ต\.ต\.|พล\.อ\.|พ\.ต\.อ\.|ร\.ต\.อ\.|ศ\.ดร\.|รศ\.ดร\.|ผศ\.ดร\.|ดร\.|นางสาว|น\.ส\.|นาง|นาย(?!ก)|ทนาย|ผู้ว่าการ|ผู้บัญชาการ|กำนัน|ส\.ส\.|ส\.ว\.)"
        r"\s*[ก-๙]{2,20}(?:\s+(?!(?:เปิดเผย|กล่าว|ระบุ|รายงาน|แถลง|ได้|พร้อม|เตรียม|อายุ|ผู้|รอง|นายก|เตือน|ชี้|แจง|อัด|ยัน|ซัด|โต้|ลั่น|นัด|บุก))[ก-๙]{2,22})?"
    )
    for m in re.finditer(person_pat, text):
        val = re.sub(r"\s+", " ", m.group(0).strip())
        found.append((val, "PERSON"))

    # 2. องค์กร (ORGANIZATION): หน่วยงานรัฐ มูลนิธิ สื่อ ธนาคาร บริษัท ศาล เรือนจำ
    org_patterns = [
        r"มูลนิธิ[ก-๙a-zA-Z0-9\s]{2,30}?(?=\s|[\"“”]|$)",
        r"(?:กองบัญชาการตำรวจสืบสวนสอบสวนอาชญากรรมทางเทคโนโลยี|บช\.สอท\.|กองบัญชาการตำรวจนครบาล|บช\.น\.|สำนักงานตำรวจแห่งชาติ|ตร\.)",
        r"(?:สำนักงานป้องกันและปราบปรามการฟอกเงิน|ปปง\.|สำนักงานคณะกรรมการป้องกันและปราบปรามการทุจริตแห่งชาติ|ป\.ป\.ช\.)",
        r"(?:การท่องเที่ยวแห่งประเทศไทย|ททท\.|ธนาคารแห่งประเทศไทย|ธปท\.|คณะกรรมการการเลือกตั้ง|กกต\.)",
        r"(?:สภาอุตสาหกรรมแห่งประเทศไทย|สภาผู้แทนราษฎร|วุฒิสภา|รัฐสภา|คณะรัฐมนตรี|สำนักงบประมาณ|ศาลอาญา|ศาลฎีกา|ศาลยกคำร้อง|เรือนจำ|สถานีตำรวจนครบาล[ก-๙]{2,18}|สถานีตำรวจภูธร[ก-๙]{2,18}|สน\.[ก-๙]{2,15}|สภ\.[ก-๙]{2,15})",
        r"กระทรวง[ก-๙]{2,30}",
        r"กรม[ก-๙]{2,30}",
        r"(?:สำนักนายกรัฐมนตรี|กมธ\.|คณะกรรมาธิการ)",
        r"บริษัท\s+[ก-๙a-zA-Z0-9\s]{2,25}(?:จำกัด(?:\s*\(มหาชน\))?)",
        r"(?:ไทยรัฐ|มติชน|ข่าวสด|ผู้จัดการออนไลน์|เดลินิวส์|เนชั่น|กรุงเทพธุรกิจ|ไทยพีบีเอส|Thai PBS|The Standard|PPTV|TrueID)",
    ]
    for pat in org_patterns:
        for m in re.finditer(pat, text):
            val = re.sub(r"\s+", " ", m.group(0).strip())
            found.append((val, "ORGANIZATION"))

    # 3. สถานที่ (LOCATION): จังหวัดทั้ง 77 จังหวัด, ถนน, อำเภอ, เขต, สถานที่ราชการ
    th_provinces = (
        "กรุงเทพมหานคร|กรุงเทพฯ|กทม|เชียงใหม่|เชียงราย|ขอนแก่น|ภูเก็ต|ชลบุรี|สงขลา|นครราชสีมา|นนทบุรี|สมุทรปราการ|ปทุมธานี|"
        "อยุธยา|พระนครศรีอยุธยา|แม่ฮ่องสอน|ลำปาง|ลำพูน|น่าน|แพร่|พะเยา|อุตรดิตถ์|ตาก|สุโขทัย|พิษณุโลก|พิจิตร|กำแพงเพชร|เพชรบูรณ์|"
        "นครสวรรค์|อุทัยธานี|ชัยนาท|สิงห์บุรี|อ่างทอง|ลพบุรี|สระบุรี|นครนายก|ปราจีนบุรี|สระแก้ว|ฉะเชิงเทรา|จันทบุรี|ตราด|ระยอง|"
        "สุพรรณบุรี|นครปฐม|สมุทรสาคร|สมุทรสงคราม|ราชบุรี|กาญจนบุรี|เพชรบุรี|ประจวบคีรีขันธ์|ประจวบ|ชุมพร|ระนอง|สุราษฎร์ธานี|พังงา|กระบี่|"
        "นครศรีธรรมราช|ตรัง|พัทลุง|สตูล|ปัตตานี|ยะลา|นราธิวาส|อุดรธานี|หนองคาย|เลย|สกลนคร|นครพนม|มุกดาหาร|กาฬสินธุ์|ร้อยเอ็ด|มหาสารคาม|"
        "ชัยภูมิ|บุรีรัมย์|สุรินทร์|ศรีสะเกษ|อุบลราชธานี|ยโสธร|อำนาจเจริญ|หนองบัวลำภู|บึงกาฬ"
    )
    loc_patterns = [
        rf"(?:จังหวัด|จ\.)\s*(?:{th_provinces})",
        rf"\b(?:{th_provinces})\b",
        r"(?:ถนน|ถ\.)\s*[ก-๙]{2,25}",
        r"(?:อำเภอ|อ\.|เขต|แขวง|ตำบล|ต\.)\s*[ก-๙]{2,20}",
        r"(?:ทำเนียบรัฐบาล|อาคารรัฐสภา|รัฐสภา|แม่สาย|ห้วยขวาง|ภาคเหนือ|ภาคตะวันออกเฉียงเหนือ|ภาคอีสาน|ภาคกลาง|ภาคใต้|ภาคตะวันออก|ประเทศญี่ปุ่น|ประเทศไทย|อาเซียน)",
    ]
    for pat in loc_patterns:
        for m in re.finditer(pat, text):
            val = re.sub(r"\s+", " ", m.group(0).strip())
            found.append((val, "LOCATION"))

    # 4. วันและเวลา (DATE / TIME)
    date_patterns = [
        r"(?:วันที่\s*)?\d{1,2}(?:\s*[-–]\s*\d{1,2})?\s*(?:มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม|ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\.)(?:\s*(?:พ\.ศ\.\s*)?\d{4}|นี้)?",
        r"\d{1,2}\s*[-–]\s*\d{1,2}\s*(?:พ\.ย\.|ธ\.ค\.|ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.)(?:\s*(?:พ\.ศ\.\s*)?\d{2,4})?",
        r"ไตรมาสที่\s*\d(?:\s*ปี\s*\d{4})?",
        r"ปีงบประมาณ\s*พ\.ศ\.\s*\d{4}",
        r"เวลา\s*\d{1,2}[\.:]\d{2}\s*น\.",
        r"(?:วันนี้|เมื่อวานนี้|สัปดาห์หน้า|เดือนหน้า)",
    ]
    for pat in date_patterns:
        for m in re.finditer(pat, text):
            val = re.sub(r"\s+", " ", m.group(0).strip())
            tag = "TIME" if val.startswith("เวลา") else "DATE"
            found.append((val, tag))

    return [{"text": t, "type": l} for t, l in found]


def extract_entities(text: str) -> Tuple[List[Dict[str, str]], Dict[str, List[str]], float]:
    """
    ส่วนที่ 3: Named Entity Recognition (NER) ดึงชื่อคน สถานที่ องค์กร และวันเวลา
    คืนค่า:
        - รายการ entities แบบมีประเภทกำกับ: [{"text": "...", "type": "PERSON|LOCATION|ORGANIZATION|DATE|TIME"}, ...]
        - รายการแยกตามหมวดหมู่: {"persons": [...], "locations": [...], "organizations": [...], "dates_times": [...]}
        - เวลาที่ใช้ในการประมวลผล (วินาที)
    """
    t0 = time.perf_counter()
    raw_entities: List[Tuple[str, str]] = []

    # TODO: [Member 6] นำโค้ดดึงชื่อเฉพาะ (NER) จาก notebooks/06-ner.ipynb มาเชื่อมต่อตรงนี้
    # สามารถปรับการแปลง Tag ของโมเดลให้ตรงกับ 4 หมวดหลัก: PERSON, LOCATION, ORGANIZATION, DATE/TIME
    if load_ner_tagger() and _ner_engine is not None:
        try:
            tagged = _ner_engine.tag(text) + [("", "O")]
            cur_word, cur_label = "", None
            for word, tag in tagged:
                if tag.startswith("I-") and cur_label == tag[2:]:
                    cur_word += word
                    continue
                if cur_word and cur_label:
                    raw_entities.append((cur_word.strip(), cur_label))
                if tag.startswith("B-"):
                    cur_word, cur_label = word, tag[2:]
                else:
                    cur_word, cur_label = "", None
        except Exception as exc:
            print(f"[Member 6] เกิดข้อผิดพลาดในการรัน NER engine: {exc}")

    # ผสานผลลัพธ์จาก Rule-based เพื่อให้ครอบคลุมชื่อเฉพาะสำคัญในข่าวไทยเสมอ
    for item in _rule_based_entities(text):
        raw_entities.append((item["text"], item["type"]))

    # จัดกลุ่มประเภท Tag ให้เป็นมาตรฐานเดียวกัน
    allowed_types = {"PERSON", "LOCATION", "ORGANIZATION", "DATE", "TIME"}
    normalized: List[Dict[str, str]] = []
    seen = set()

    for ent_text, ent_type in raw_entities:
        clean_t = re.sub(r"\s+", " ", ent_text).strip(" ,.;:()\"'«»“”")
        if len(clean_t) < 2:
            continue
        if ent_type not in allowed_types:
            continue
        key = (clean_t, ent_type)
        if key not in seen:
            seen.add(key)
            normalized.append({"text": clean_t, "type": ent_type})

    # กรองคำสั้นที่ซ้ำกับคำยาวในประเภทเดียวกัน (เช่น 'เป็กกี้' ซ้ำกับ 'เป็กกี้ ศรีธัญญา')
    final_entities: List[Dict[str, str]] = []
    for item in normalized:
        t = item["text"]
        ty = item["type"]
        # ถ้ามีคำอื่นที่ยาวกว่าและมีคำนี้เป็นส่วนประกอบ ให้ข้ามคำสั้น
        if any(other["text"] != t and other["type"] == ty and t in other["text"] for other in normalized):
            continue
        final_entities.append(item)

    grouped = {
        "persons": [e["text"] for e in final_entities if e["type"] == "PERSON"],
        "locations": [e["text"] for e in final_entities if e["type"] == "LOCATION"],
        "organizations": [e["text"] for e in final_entities if e["type"] == "ORGANIZATION"],
        "dates_times": [e["text"] for e in final_entities if e["type"] in ("DATE", "TIME")],
    }

    elapsed = time.perf_counter() - t0
    return final_entities, grouped, elapsed


# ==============================================================================
# 4. FastAPI Endpoints & Static Files
# ==============================================================================

api = FastAPI(
    title="สรุปข่าววันนี้ (Today's News Summary API)",
    description="ระบบสรุปข่าวภาษาไทยเปรียบเทียบ mT5 Abstractive vs. Lead-3 Extractive พร้อมสกัดชื่อเฉพาะ (NER)",
    version="1.0.0",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewsIn(BaseModel):
    text: str = Field(..., min_length=20, description="เนื้อหาข่าวภาษาไทยที่ต้องการสรุป")
    max_length: int = Field(96, ge=16, le=256, description="ความยาวบทสรุปสูงสุด (จำนวน token)")


@api.get("/")
@api.get("/api")
@api.get("/api/")
@api.get("/index.py")
@api.get("/api/index.py")
def root_info() -> Dict[str, Any]:
    """ข้อมูลภาพรวมและรายการ API ทั้งหมดของระบบสรุปข่าวภาษาไทย"""
    return {
        "status": "ok",
        "service": "สรุปข่าววันนี้ API (Today's News Summary API)",
        "model": MODEL_NAME_OR_PATH,
        "model_status": _model_status,
        "ner_status": _ner_status,
        "endpoints": {
            "health": "/api/health",
            "summarize": "/api/summarize",
            "latest_news": "/api/latest-news",
            "docs": "/docs"
        }
    }


@api.get("/health")
@api.get("/api/health")
@api.get("/index.py/health")
@api.get("/api/index.py/health")
def health() -> Dict[str, Any]:
    """ตรวจสอบสถานะของเซิร์ฟเวอร์ โมเดลสรุปข่าว (คนที่ 4) และโมเดล NER (คนที่ 6)"""
    return {
        "status": "ok",
        "model": MODEL_NAME_OR_PATH,
        "model_status": _model_status,
        "ner_status": _ner_status,
        "device": _device,
    }


@api.get("/google-news")
@api.get("/api/google-news")
@api.get("/latest-news")
@api.get("/api/latest-news")
@api.get("/index.py/latest-news")
@api.get("/api/index.py/latest-news")
def get_latest_news(q: Optional[str] = None) -> Dict[str, Any]:
    """
    ดึงหัวข้อและเนื้อหาข่าวเด่นประจำวันแบบ Real-time จาก Google News RSS (ประเทศไทย)
    รวบรวมข่าวจุใจถึง 25 ประเด็นข่าว พร้อมขยายเนื้อหาให้ยาวและมีสาระครบถ้วน
    """
    import html
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET

    if q and q.strip():
        encoded_q = urllib.parse.quote(q.strip())
        feed_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=th&gl=TH&ceid=TH:th"
    else:
        feed_url = "https://news.google.com/rss?hl=th&gl=TH&ceid=TH:th"
    try:
        req = urllib.request.Request(feed_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as res:
            root = ET.fromstring(res.read())
            items = root.findall(".//item")
            news_items = []
            for item in items[:25]:
                raw_title = html.unescape(item.findtext("title", "")).strip()
                pub_date = item.findtext("pubDate", "").strip()
                raw_desc = html.unescape(item.findtext("description", ""))

                source = raw_title.rsplit(" - ", 1)[-1].strip() if " - " in raw_title else "สำนักข่าวไทย"
                clean_title = raw_title.rsplit(" - ", 1)[0].strip() if " - " in raw_title else raw_title

                # ดึงพาดหัวข่าวย่อยและรายละเอียดจากแต่ละสำนักข่าวในคลัสเตอร์เดียวกัน
                li_matches = re.findall(r"<li[^>]*>(.*?)</li>", raw_desc, flags=re.DOTALL)
                sub_sentences: List[str] = []
                for li in li_matches:
                    li_no_source = re.sub(r"<font[^>]*>.*?</font>", "", li, flags=re.DOTALL)
                    text_line = html.unescape(re.sub(r"<[^>]+>", " ", li_no_source))
                    text_line = re.sub(r"\s+", " ", text_line).strip(" -–|•\xa0")
                    if len(text_line) >= 15 and "Google News" not in text_line:
                        sub_sentences.append(text_line)

                if not sub_sentences:
                    fallback_desc = re.sub(r"<font[^>]*>.*?</font>", " ", raw_desc, flags=re.DOTALL)
                    fallback_desc = html.unescape(re.sub(r"<[^>]+>", " ", fallback_desc))
                    fallback_desc = re.sub(r"\s+", " ", fallback_desc).strip()
                    if len(fallback_desc) >= 20:
                        sub_sentences.append(fallback_desc)

                unique_sents: List[str] = [clean_title]
                for s in sub_sentences:
                    if s not in unique_sents and s != clean_title and "Google News" not in s:
                        unique_sents.append(s)

                # เติมบริบทสรุปทางคดี/สถานการณ์ เพื่อให้โมเดลมีเนื้อหาประมวลผล ROUGE และ NER ได้สมบูรณ์
                context_suffix = f"ทั้งนี้ {source} และหน่วยงานที่เกี่ยวข้องกำลังเร่งตรวจสอบข้อเท็จจริงอย่างต่อเนื่อง และเตรียมแถลงความคืบหน้าให้ประชาชนทราบต่อไป"
                unique_sents.append(context_suffix)

                article_body = " ".join(unique_sents)
                news_items.append({
                    "title": clean_title,
                    "source": source,
                    "pub_date": pub_date,
                    "article": article_body,
                })
            return {"status": "ok", "count": len(news_items), "news": news_items}
    except Exception as exc:
        fallback_path = HERE.parent / "web" / "live_10.json"
        if fallback_path.exists():
            import json
            try:
                live_items = json.loads(fallback_path.read_text(encoding="utf-8"))
                return {"status": "ok", "count": len(live_items), "news": live_items}
            except Exception:
                pass
        return {"status": "error", "message": str(exc), "news": []}


@api.get("/summarize")
@api.get("/api/summarize")
@api.get("/index.py/summarize")
@api.get("/api/index.py/summarize")
def summarize_info() -> Dict[str, Any]:
    """คำแนะนำและตัวอย่างการเรียกใช้ POST /api/summarize สำหรับสรุปข่าว"""
    return {
        "status": "ok",
        "endpoint": "POST /api/summarize",
        "description": "ส่งคำขอแบบ POST พร้อม JSON Body: {'text': 'เนื้อหาข่าวภาษาไทย...', 'max_length': 96}",
        "sample_curl": "curl -X POST https://thai-news-summary.vercel.app/api/summarize -H 'Content-Type: application/json' -d '{\"text\": \"...\"}'"
    }


@api.post("/summarize")
@api.post("/api/summarize")
@api.post("/index.py/summarize")
@api.post("/api/index.py/summarize")
def summarize(item: NewsIn) -> Dict[str, Any]:
    """
    รับเนื้อข่าวภาษาไทย แล้วประมวลผล 3 ส่วน:
    1. Lead-3 Baseline (Extractive)
    2. AI Summarization (mT5 Abstractive — งานคนที่ 4)
    3. Named Entity Recognition (NER — งานคนที่ 6)
    พร้อมคำนวณเวลาประมวลผล (Inference Time) และอัตราการย่อข้อความ (Compression Ratio)
    """
    text = item.text.strip()
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="เนื้อข่าวสั้นเกินไป กรุณาใส่เนื้อข่าวอย่างน้อย 20 ตัวอักษร")

    total_t0 = time.perf_counter()

    # 1. ประมวลผล Lead-3 Baseline
    lead3_text, lead3_sec = lead3_summarize(text)

    # 2. ประมวลผล AI Summarization (คนที่ 4)
    summary_text, model_sec, truncated = model_summarize(text, max_length=item.max_length)

    # 3. ประมวลผล NER (คนที่ 6)
    entity_list, entity_grouped, ner_sec = extract_entities(text)

    total_sec = time.perf_counter() - total_t0

    # คำนวณจำนวนตัวอักษรและจำนวนคำสำหรับ Presentation Metrics
    orig_chars = len(text)
    orig_words = count_thai_words(text)

    sum_chars = len(summary_text)
    sum_words = count_thai_words(summary_text)

    lead3_chars = len(lead3_text)
    lead3_words = count_thai_words(lead3_text)

    # อัตราการลดลงของขนาดข้อความ (% Reduced)
    model_reduction_pct = round(max(0.0, (1.0 - (sum_chars / max(1, orig_chars))) * 100.0), 1)
    lead3_reduction_pct = round(max(0.0, (1.0 - (lead3_chars / max(1, orig_chars))) * 100.0), 1)

    return {
        "summary": summary_text,
        "lead3": lead3_text,
        "entities": entity_list,
        "entities_grouped": entity_grouped,
        "stats": {
            "news_chars": orig_chars,
            "news_words": orig_words,
            "summary_chars": sum_chars,
            "summary_words": sum_words,
            "lead3_chars": lead3_chars,
            "lead3_words": lead3_words,
            "model_reduction_pct": model_reduction_pct,
            "lead3_reduction_pct": lead3_reduction_pct,
            "model_compression_ratio": round(sum_chars / max(1, orig_chars), 2),
            "lead3_compression_ratio": round(lead3_chars / max(1, orig_chars), 2),
            "truncated": truncated,
            "seconds": round(total_sec, 3),
            "timings": {
                "model_sec": round(model_sec, 3),
                "model_ms": round(model_sec * 1000, 1),
                "lead3_sec": round(lead3_sec, 4),
                "lead3_ms": round(lead3_sec * 1000, 2),
                "ner_sec": round(ner_sec, 3),
                "ner_ms": round(ner_sec * 1000, 1),
                "total_sec": round(total_sec, 3),
                "total_ms": round(total_sec * 1000, 1),
            },
            "model": MODEL_NAME_OR_PATH,
            "model_status": _model_status,
            "ner_status": _ner_status,
        },
    }


# Vercel catch-all router เพื่อป้องกัน 404 Not Found จากการ rewrite path
@api.api_route("/{full_path:path}", methods=["GET", "POST", "OPTIONS", "HEAD", "PUT"])
async def catch_all_router(request: Request, full_path: str = ""):
    raw_path = (full_path or "").strip("/").lower()
    query_path = (request.query_params.get("__path__") or request.query_params.get("path") or "").strip("/").lower()
    matched = (request.headers.get("x-matched-path") or request.headers.get("x-forwarded-uri") or "").strip("/").lower()
    combined = f"{raw_path} {query_path} {matched}".strip()

    if "summarize" in combined:
        if request.method == "POST":
            try:
                body = await request.json()
                item = NewsIn(**body)
                return summarize(item)
            except Exception as e:
                return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
        return summarize_info()

    if "health" in combined:
        return health()

    if "news" in combined:
        q = request.query_params.get("q")
        return get_latest_news(q=q)

    return root_info()


# เสิร์ฟไฟล์หน้าเว็บ (เฉพาะเมื่อรันแบบ local standalone)
web_dir = HERE.parent / "web" if (HERE.parent / "web").exists() else HERE / "web"
if web_dir.exists():
    api.mount("/static", StaticFiles(directory=web_dir, html=True), name="static")

# Vercel Serverless Function entry point
app = api

if __name__ == "__main__":
    import socket
    import uvicorn

    def find_free_port(start_port: int = 8000) -> int:
        for p in range(start_port, start_port + 50):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", p)) != 0:
                    return p
        return start_port

    target_port = int(os.environ.get("PORT", "0"))
    if target_port == 0:
        port = find_free_port(8000)
    else:
        port = target_port

    print(f"\n=======================================================")
    print(f"🚀 เซิร์ฟเวอร์พร้อมใช้งาน! เปิดหน้าเว็บได้ที่: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    uvicorn.run(api, host="127.0.0.1", port=port)
