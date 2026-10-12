# ML-08 Deep Convolutional Neural Network (DCNN) - Ramen Ratings

## ข้อมูล

[Ramen Ratings (Kaggle)](https://www.kaggle.com/datasets/residentmario/ramen-ratings) ไฟล์ `ramen-ratings.csv` มี 2,580 แถว 7 คอลัมน์

| คอลัมน์ | ความหมาย |
|---|---|
| `Review #` | ลำดับรีวิว |
| `Brand` | ยี่ห้อ |
| `Variety` | ชื่อรสหรือชนิดสินค้า |
| `Style` | รูปแบบบรรจุภัณฑ์ (Pack, Bowl, Cup, Tray, ...) |
| `Country` | ประเทศ |
| `Stars` | คะแนน 0-5 (บางแถวเป็น `Unrated`) |
| `Top Ten` | ติดอันดับ Top Ten ของปีหรือไม่ |

## งานที่โมเดลทำ

โมเดลอ่านข้อความ `Brand | Variety | Country` แล้วทำนายคลาส เลือกเป้าหมายได้ที่ตัวแปร `TARGET` ใน `main.py`

| `TARGET` | ทำนายอะไร | คลาส |
|---|---|---|
| `"Style"` (ค่าเริ่มต้น) | รูปแบบบรรจุภัณฑ์ | Bowl, Cup, Pack, Tray |
| `"Rating"` | ระดับคะแนนจาก `Stars` | Low (< 3), Medium (3 - 3.99), High (>= 4) |

คลาสที่มีน้อยกว่า `MIN_PER_CLASS` (30 แถว) จะถูกตัดออก เช่น Box, Can, Bar

## โครงสร้างโปรเจกต์

```text
ML-08-DCNN/
│
├── data/
│   └── ramen-ratings.csv
│
├── mini-proj/
│   ├── data_loader.py     # อ่าน CSV, ทำความสะอาด, สร้างข้อความและป้ายกำกับ
│   ├── evaluate.py        # accuracy, classification report, confusion matrix, กราฟ history
│   ├── main.py            # สคริปต์หลัก รันทุกขั้นตอน
│   ├── preprocessing.py   # แปลงข้อความเป็นตัวพิมพ์เล็ก, เข้ารหัสทีละตัวอักษร, เติม padding
│   ├── split_data.py      # แบ่ง train / validation / test แบบ stratified
│   ├── test_vgg.py        # สุ่มทำนาย 4 แถวจากชุดทดสอบ
│   └── vgg_model.py       # สร้าง, เทรน, บันทึก และทำนายด้วยโมเดล 1-D VGG
│   │
│   └── outputs/           # ไฟล์ที่สร้างขึ้นหลังรัน (โมเดล, กราฟ, ข้อมูลที่แบ่งแล้ว)
│       ├── classes.json
│       ├── cinfusion_matrix.png
│       ├── history.json
│       ├── prediction_sample.png
│       └── training_history.png
│
└── requirements.txt
```

## ขั้นตอนการทำงาน

1. **โหลดข้อมูล** อ่าน CSV, ตัดแถวที่ไม่มีค่า (`Style` ว่าง หรือ `Stars` เป็น `Unrated`), ตัดคลาสที่มีน้อยเกินไป
2. **เตรียมข้อมูล** แปลงข้อความเป็นตัวเลขทีละตัวอักษร ความยาวคงที่ 96 ตัวอักษร (ครอบคลุมประมาณ 99% ของแถว)
3. **แบ่งชุดข้อมูล** train 60% / validation 20% / test 20% โดยรักษาสัดส่วนคลาส
4. **เทรนโมเดล** ใช้ class weight ชดเชยคลาสที่ไม่สมดุล และ EarlyStopping เลือกน้ำหนักรอบที่ดีที่สุด
5. **ทำนาย** บนชุดทดสอบ
6. **ประเมินผล** แสดง accuracy, precision/recall/F1 และบันทึก confusion matrix กับกราฟการเทรน

## สถาปัตยกรรมโมเดล

```text
Input (ตัวอักษร 96 ตัว)
  → Embedding (32 มิติ) → SpatialDropout
  → บล็อก 1: [Conv1D 32 → BatchNorm] ×2 → MaxPool
  → บล็อก 2: [Conv1D 64 → BatchNorm] ×2 → MaxPool
  → บล็อก 3: [Conv1D 128 → BatchNorm] ×3 → MaxPool
  → GlobalMaxPooling
  → Dense 128 → Dropout → Dense 64 → Dropout
  → Softmax (จำนวนคลาส)
```

มี 2 ขนาดให้เลือกใน `main.py`: `VGG_1D` (7 ชั้น Conv) และ `VGG_SMALL` (4 ชั้น Conv เบากว่า)

## วิธีติดตั้ง

ต้องใช้ Python 3.10 - 3.12

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

ไลบรารีที่ใช้: `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`

## วิธีรัน

วางไฟล์ `ramen-ratings.csv` ไว้ที่ `data/ramen-ratings.csv` (อยู่ข้างโฟลเดอร์ `mini-proj/`) แล้วรัน

```bash
cd mini-proj
python main.py        # เทรนและประเมินผล
python test_vgg.py    # สุ่มทำนาย 9 แถว (ต้องรัน main.py ให้เสร็จก่อน)
```

## ค่าที่ปรับได้ใน `main.py`

| ตัวแปร | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `TARGET` | `"Style"` | เป้าหมายการทำนาย: `"Style"` หรือ `"Rating"` |
| `MIN_PER_CLASS` | 30 | ตัดคลาสที่มีจำนวนแถวน้อยกว่านี้ |
| `MAX_LEN` | 96 | ความยาวข้อความ (จำนวนตัวอักษร) |
| `TEST_SIZE`, `VAL_SIZE` | 0.2, 0.2 | สัดส่วนชุดทดสอบและชุดตรวจสอบ |
| `EPOCHS` | 50 | จำนวนรอบสูงสุดในการเทรน |
| `BATCH_SIZE` | 32 | ขนาด batch |
| `BLOCKS` | `VGG_1D` | ขนาดโมเดล: `VGG_1D` หรือ `VGG_SMALL` |
| `SEED` | 42 | ค่า random seed เพื่อให้ผลซ้ำได้ |

หากเปลี่ยน `TARGET` ต้องรัน `main.py` ใหม่ก่อนรัน `test_vgg.py` เสมอ

## ไฟล์ที่ได้ใน `outputs/`

| ไฟล์ | รายละเอียด |
|---|---|
| `vgg_model.keras` | โมเดลที่เทรนแล้ว |
| `history.json` | ค่า loss/accuracy ของแต่ละ epoch |
| `confusion_matrix.png` | ตาราง confusion matrix |
| `training_history.png` | กราฟ accuracy และ loss ระหว่างเทรน |
| `prediction_sample.png` | ภาพตัวอย่างการทำนายจาก `test_vgg.py` |
| `classes.json`, `*.npy` | รายชื่อคลาสและข้อมูลที่แบ่งแล้ว (ใช้โดย `test_vgg.py`) |

## ผลลัพธ์ (เป้าหมาย Style, SEED = 42)

| คลาส | precision | recall | F1 |
|---|---|---|---|
| Bowl | 0.59 | 0.55 | 0.57 |
| Cup | 0.71 | 0.67 | 0.69 |
| Pack | 0.79 | 0.83 | 0.81 |
| Tray | 0.79 | 0.68 | 0.73 |

- accuracy บนชุดทดสอบ **ประมาณ 74%**, macro F1 ประมาณ **0.70**
- ทดสอบกับ seed อื่น (7, 123) ได้ accuracy ประมาณ 69 - 73% ผลจึงแปรผันได้ตาม seed และเครื่องที่รัน
- คลาสที่ยากที่สุดคือ **Bowl** เพราะชื่อสินค้าอย่างเดียวบอกรูปแบบบรรจุภัณฑ์ได้ไม่เสมอไป

## ข้อสังเกตและข้อจำกัด

- ข้อมูลมีเพียงประมาณ 2,500 แถว จึงเกิด **overfitting** ได้ง่าย (train accuracy สูงกว่า validation ประมาณ 18%) เหมาะกับการเรียนรู้ ไม่ใช่โมเดลสำหรับใช้งานจริง
- ช่วงแรกของการเทรน validation accuracy อาจค้างที่สัดส่วนของคลาสใหญ่ (โมเดลยังทายคลาสเดียว) แล้วจะเริ่มเรียนรู้จริงหลังผ่านไปประมาณ 10 epoch ถือเป็นเรื่องปกติ
- ควรดู **macro F1** ควบคู่กับ accuracy เสมอ เพราะคลาสไม่สมดุล (Pack มากกว่า Tray หลายเท่า)
- หากต้องการลด overfitting ให้ลองใช้ `VGG_SMALL`, เพิ่มค่า Dropout หรือลด `EPOCHS`
