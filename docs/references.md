# เอกสารอ้างอิง · ระบบสรุปข่าวภาษาไทย

แยกตามงานของแต่ละคน ทุกรายการเช็กแล้วว่ามีอยู่จริง (ดูลิงก์) · รูปแบบอ้างอิงด้านล่างเป็น APA ย่อ ปรับตามที่อาจารย์กำหนดได้

---

## ข้อมูล (คนที่ 1)

**[1] ThaiSum** ชุดข้อมูลหลัก
Chumpolsathien, N. (2020). *ThaiSum: The largest dataset for Thai text summarization* [Dataset]. GitHub / Hugging Face.
- ข่าวพร้อมบทสรุปที่นักข่าวเขียน 350,000+ คู่ จากไทยรัฐ ไทยพีบีเอส ประชาไท The Standard
- ตัดข่าวที่สั้นกว่า 150 คำ หรือบทสรุปสั้นกว่า 15 คำ ออกไปแล้ว · ผู้สร้างลอง BertSum/ARedSum เป็น baseline ไว้ด้วย
- ที่มาของงาน: วิทยานิพนธ์ปริญญาโท Beijing Institute of Technology (2020)
- 🔗 https://github.com/nakhunchumpolsathien/ThaiSum · https://huggingface.co/datasets/pythainlp/thaisum

**[2] PyThaiNLP** ใช้ตัดคำ ทำความสะอาด และ NER
Phatthiyaphaibun, W., Chaovavanich, K., Suriyawongkul, A., et al. (2023). PyThaiNLP: Thai natural language processing in Python. In *Proceedings of the 3rd Workshop for NLP Open Source Software (NLP-OSS 2023)* (pp. 25–36).
- 🔗 https://arxiv.org/abs/2312.04649 · https://github.com/PyThaiNLP/pythainlp

---

## Extractive summarization (คู่ B · คนที่ 2, 3)

**[3] Luhn (1958)** ต้นกำเนิดการสรุปอัตโนมัติ: ให้คะแนนประโยคจากคำสำคัญ แล้วเลือกประโยคคะแนนสูง (แนวคิดเดียวกับ TF-IDF sentence scoring)
Luhn, H. P. (1958). The automatic creation of literature abstracts. *IBM Journal of Research and Development, 2*(2), 159–165.
- 🔗 https://courses.ischool.berkeley.edu/i256/f06/papers/luhn58.pdf

**[4] TextRank** (คนที่ 2) สร้างกราฟประโยค เชื่อมประโยคที่คล้ายกัน แล้วจัดอันดับแบบ PageRank ไม่ต้องเทรน
Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing order into text. In *Proceedings of EMNLP 2004* (pp. 404–411).
- 🔗 https://aclanthology.org/W04-3252/

**[5] Lead-3 baseline** (คนที่ 3) เอา 3 ประโยคแรกของข่าวเป็นบทสรุป — ข่าวมักเขียนเรื่องสำคัญไว้ต้นข่าว จึงเป็น baseline ที่แข็งมาก ในงานนี้ โมเดลของผู้เขียนยังเอาชนะ Lead-3 ไม่ได้
See, A., Liu, P. J., & Manning, C. D. (2017). Get to the point: Summarization with pointer-generator networks. In *Proceedings of ACL 2017* (pp. 1073–1083).
- 🔗 https://arxiv.org/abs/1704.04368
- ⚠️ ข่าวไทยใช้ช่องว่างแทนจุดจบประโยค ต้องตัดประโยคด้วย `pythainlp.tokenize.sent_tokenize` ก่อน

---

## Abstractive summarization (คู่ C · คนที่ 4, 5)

**[6] mT5** (คนที่ 4 · โมเดลหลัก) T5 แบบหลายภาษา เทรนบน mC4 101 ภาษา (มีภาษาไทย) ทุกงานเป็นรูปแบบ "ข้อความเข้า → ข้อความออก" จึงใช้สรุปได้ตรง ๆ
Xue, L., Constant, N., Roberts, A., Kale, M., Al-Rfou, R., Siddhant, A., Barua, A., & Raffel, C. (2021). mT5: A massively multilingual pre-trained text-to-text transformer. In *Proceedings of NAACL 2021* (pp. 483–498).
- 🔗 https://aclanthology.org/2021.naacl-main.41 · โมเดล: `google/mt5-small`, `google/mt5-base`

**[7] XL-Sum** ตัวอย่างงานที่ fine-tune mT5 สรุปข่าวหลายภาษา (มีภาษาไทย) ใช้เป็นงานอ้างอิง/เทียบผลได้ และมีโมเดลที่เทรนแล้วให้ลองเป็นจุดเริ่ม
Hasan, T., Bhattacharjee, A., Islam, M. S., Mubasshir, K., Li, Y.-F., Kang, Y.-B., Rahman, M. S., & Shahriyar, R. (2021). XL-Sum: Large-scale multilingual abstractive summarization for 44 languages. In *Findings of ACL-IJCNLP 2021* (pp. 4693–4703).
- 🔗 https://arxiv.org/abs/2106.13822 · https://github.com/csebuetnlp/xl-sum · โมเดล: `csebuetnlp/mT5_multilingual_XLSum`
- ⚠️ ข้อมูล XL-Sum เป็น CC BY-NC-SA 4.0 (ใช้เพื่อการศึกษาได้)

**[8] mBART** (คนที่ 5 · ตัวเทียบ) seq2seq ที่ pre-train โดยให้ซ่อมข้อความที่ถูกทำให้เสีย (denoising) หลายภาษา · ใช้รุ่น mBART-50 ที่มีภาษาไทย (`th_TH`)
Liu, Y., Gu, J., Goyal, N., Li, X., Edunov, S., Ghazvininejad, M., Lewis, M., & Zettlemoyer, L. (2020). Multilingual denoising pre-training for neural machine translation. *Transactions of the ACL, 8*, 726–742.
- 🔗 https://arxiv.org/abs/2001.08210 · โมเดล: `facebook/mbart-large-50`

---

## NER (คนที่ 6)

ใช้ **[2] PyThaiNLP** (`pythainlp.tag.NER("thainer")`) ดึงชื่อเฉพาะ แล้วเทียบว่าชื่อสำคัญในข่าวต้นฉบับ ไปโผล่ในบทสรุปไหม และบทสรุปมีชื่อที่ไม่อยู่ในข่าวไหม (สัญญาณของการแต่งข้อมูล — ดู [10])

---

## การวัดผล (คนที่ 7)

**[9] ROUGE** ตัววัดหลัก นับ n-gram ที่ซ้อนกันระหว่างบทสรุปของโมเดลกับของคน (ROUGE-1 = คำเดี่ยว, ROUGE-2 = คู่คำ, ROUGE-L = ลำดับร่วมที่ยาวที่สุด)
Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. In *Text Summarization Branches Out (ACL 2004 Workshop)* (pp. 74–81).
- 🔗 https://aclanthology.org/W04-1013/
- ⚠️ ไลบรารี ROUGE ทั่วไปแยกคำด้วยช่องว่าง — ภาษาไทยต้องตัดคำด้วย PyThaiNLP ก่อน แล้วเว้นวรรคคั่น

**[10] ความถูกต้องของบทสรุป (hallucination)** โมเดล abstractive ทุกตัวในงานนี้มีเนื้อหาที่ข่าวต้นฉบับไม่ได้พูด · โมเดล pre-trained ดีกว่าโมเดลอื่นทั้ง ROUGE และความถูกต้อง · ROUGE อย่างเดียววัดความถูกต้องไม่ได้ ต้องให้คนอ่านด้วย → ใช้เป็นเหตุผลของ error analysis
Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). On faithfulness and factuality in abstractive summarization. In *Proceedings of ACL 2020* (pp. 1906–1919).
- 🔗 https://aclanthology.org/2020.acl-main.173 · https://arxiv.org/abs/2005.00661

---

## ตารางสรุป: ใครอ้างอะไร

| คน | งาน | อ้างอิง |
|---|---|---|
| 1 | ข้อมูล | [1] [2] |
| 2 | TextRank / TF-IDF | [3] [4] |
| 3 | Lead-3 | [5] |
| 4 | mT5 | [6] [7] |
| 5 | mBART-50 | [8] [6] |
| 6 | NER | [2] [10] |
| 7 | ROUGE + วิเคราะห์ | [9] [10] |
| 8 | เว็บแอป | — (อ้าง [6] ว่าใช้โมเดลอะไร) |
