# Ramen Rating Classifier (CNN)

โปรเจกต์นี้ใช้ CNN แบบ 1D จำแนกว่ารีวิวบะหมี่กึ่งสำเร็จรูปแต่ละรายการเป็น
`Good` (ให้ดาว >= 4.0) หรือ `Ordinary` (ต่ำกว่านั้น) โดยดูจากข้อความที่รวมมาจาก
คอลัมน์ `Brand`, `Variety`, `Style`, `Country` เข้าด้วยกันเป็นประโยคเดียว เช่น
`"nissin cup noodles seafood pack japan"` ข้อมูลที่ใช้มาจากชุดข้อมูล
[Ramen Ratings](https://www.kaggle.com/datasets/residentmario/ramen-ratings)
บน Kaggle

## Overview

โมเดลอ่านข้อความที่รวมคอลัมน์ไว้แล้ว แปลงเป็นลำดับตัวเลข (token id) ตาม
vocabulary ที่สร้างขึ้นเอง แล้วให้ CNN แบบ 1 มิติ (Embedding → Conv1D 3 ชั้น →
GlobalMaxPooling1D → Dense) เรียนรู้รูปแบบคำที่บ่งบอกถึงคุณภาพของบะหมี่ ก่อน
ส่งต่อไปยัง Dense layer สุดท้ายเพื่อจำแนกเป็น 2 คลาส

จุดสำคัญของ pipeline นี้คือ **แบ่งข้อมูลก่อนสร้าง vocabulary**: `split_data.py`
จะทำงานบนข้อความดิบก่อน จากนั้น `preprocessing.py` ถึงจะสร้าง vocabulary จาก
เฉพาะชุด training เท่านั้น เพื่อไม่ให้คำศัพท์จากชุด validation/test รั่วไหลเข้าไป
ในโมเดล

## Project structure

```text
LAB07/
├── m-project/
│   ├── data/
│   │   └── ramen-ratings.csv       # ต้องดาวน์โหลดเอง (ดูหัวข้อ Get the data)
│   │ 
│   └── outputs/                      # ถูกสร้างตอนรันครั้งแรก
│   │   ├── classes.json
│   │   ├── confusion_matrix.png
│   │   ├── history.json
│   │   ├── prediction_sample.png
│   │   ├── text_test.json             # ข้อความดิบของชุด test (ให้ test_cnn.py แสดงผล)
│   │   ├── training_history.png
│   │   └── vocabulary.json            # vocabulary ที่สร้างจากชุด training
│   │
│   ├── cnn_model.py                  # build, train, save, predict ด้วย CNN
│   ├── data_loader.py               # อ่าน CSV, รวมคอลัมน์เป็นข้อความ, ติด label Good/Ordinary
│   ├── evaluate.py                   # accuracy, report, confusion matrix, curves
│   ├── main.py                       # รัน pipeline ทั้งหมด
│   ├── preprocessing.py              # clean ข้อความ, สร้าง vocabulary, แปลงเป็น sequence ความยาวคงที่
│   ├── split_data.py                 # stratified train/val/test split (ทำก่อนสร้าง vocabulary)
│   └── test_cnn.py                   # เช็คตัวอย่างการทำนาย (แสดงผลเป็น text panel)
│
├── README.md
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Get the data

1. ดาวน์โหลด `ramen-ratings.csv` จาก
   https://www.kaggle.com/datasets/residentmario/ramen-ratings
   (ต้องล็อกอิน Kaggle ฟรี)
2. นำไปวางไว้ที่ `ramen-cnn/data/ramen-ratings.csv`

## Run it

```bash
python main.py       # train โมเดล แล้วบันทึกผลไว้ที่ outputs/
python test_cnn.py   # สุ่มตัวอย่าง test มาโชว์ผลการทำนาย
```

## Notes / things worth knowing for a report

- เกณฑ์แบ่งคลาส: ดาว >= 4.0 = `Good`, น้อยกว่านั้น = `Ordinary` (`GOOD_THRESHOLD`
  ใน `data_loader.py`) แถวที่ไม่มีคะแนน (unrated) จะถูกตัดทิ้ง
- vocabulary ถูกสร้างจากชุด training เท่านั้น (ไม่ใช่ทั้งชุดข้อมูล) เพราะ
  `split_data.py` ทำงานก่อน `preprocessing.py` เสมอ — ถ้าสลับลำดับจะทำให้คำจาก
  ชุด test รั่วเข้าไปในโมเดล
- ความยาวข้อความถูกตัด/เติมให้เท่ากับ `MAX_LEN = 24` token และจำกัดขนาด
  vocabulary ไว้ที่ `MAX_WORDS = 10000` คำ
- `EarlyStopping` ถูกคอมเมนต์ปิดไว้ใน `cnn_model.py` ตอนนี้โมเดลจะ train ครบ
  ทุก epoch (`EPOCHS = 50`) เสมอ ถ้าเห็นสัญญาณ overfit จาก
  `training_history.png` ให้พิจารณาเปิดกลับมาใช้
- `BATCH_SIZE = 8` ค่อนข้างเล็ก ทำให้ training ช้าลงแต่ update บ่อยขึ้น ปรับเพิ่ม
  ได้ถ้าต้องการความเร็ว
- baseline ที่ต้องเอาชนะให้ได้คือการทายคลาสที่มีจำนวนมากกว่าเสมอ (`Good` หรือ
  `Ordinary`) ลองดูจำนวนแต่ละคลาสจาก output ของ `data_loader.py` เพื่อคำนวณว่า
  baseline accuracy ของชุดข้อมูลที่คุณมีอยู่คือเท่าไร
