# ============================================================
# ใบงานที่ 2: Data Preprocessing
# Dataset: Microsoft (MSFT) Stock Data: 1986 to Present (Kaggle)
# ไฟล์นี้แบ่งเป็น cell ด้วย "# %%" เปิดด้วย VS Code (Python/Jupyter extension)
# หรือ Jupyter Notebook แล้วรันทีละ cell ตามลำดับบนลงล่าง
#
# หมายเหตุ: ไฟล์ CSV ของแต่ละแหล่งบน Kaggle อาจตั้งชื่อคอลัมน์ไม่เหมือนกันทุกตัว
# (เช่น "Adj Close" อาจไม่มีในบางไฟล์) ให้เช็คด้วย df.columns หลังโหลดข้อมูล
# แล้วปรับชื่อคอลัมน์ในสคริปต์นี้ให้ตรงกับไฟล์ที่ดาวน์โหลดมาจริง
# ============================================================

# %% [Setup] Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.max_columns", None)

# ============================================================
# LAB 1: Dataset Exploration
# ============================================================

# %% LAB1.1: Load Dataset
df = pd.read_csv("MSFT_price.csv")
print("โหลดข้อมูลสำเร็จ")
print("คอลัมน์ในไฟล์:", list(df.columns))
df.head()

# %% LAB1.2: Display Shape
print("Shape (rows, columns):", df.shape)

# %% LAB1.3: Display Data Types
print(df.dtypes)
# สังเกต: คอลัมน์ Date มักถูกอ่านเป็น object (string) ต้องแปลงเป็น datetime ทีหลัง

# %% LAB1.4: Display Summary Statistics
df.describe(include="all").T

# %% LAB1.5: Display Missing Values
missing = df.isnull().sum()
print("จำนวนค่าที่หายไปต่อคอลัมน์:")
print(missing[missing > 0] if missing.sum() > 0 else "ไม่มีค่า NaN โดยตรง (แต่ต้องเช็คช่องว่างวันที่/ค่าผิดปกติเพิ่มเติม)")

# %% LAB1.6: Display Duplicate Records
print("จำนวนแถวที่ซ้ำกันทั้งแถว:", df.duplicated().sum())
print("จำนวนวันที่ (Date) ที่ซ้ำกัน:", df["Date"].duplicated().sum())

# %% LAB1.7: Display Class Distribution
# ข้อมูลหุ้นไม่มี target แบบหมวดหมู่ให้ตรงตัว จึงสร้างฟีเจอร์ "Trend"
# (ราคาปิดวันนี้สูงกว่าวันก่อนหน้าหรือไม่) มาใช้แทนเพื่อดูการกระจายตัวของ class
df_sorted = df.sort_values("Date")
trend_preview = np.where(df_sorted["Close"].diff() > 0, "Up", "Down")
print(pd.Series(trend_preview).value_counts())

# ============================================================
# LAB 2: Data Visualization
# ============================================================

# %% LAB2.1: Histogram ของฟีเจอร์ตัวเลขทั้งหมด
df.hist(figsize=(12, 10), bins=30)
plt.suptitle("Histogram of Numeric Features", y=1.02)
plt.tight_layout()
plt.show()

# %% LAB2.2: Correlation Heatmap
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(7, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.show()
# คาดว่า Open/High/Low/Close/Adj Close จะมีความสัมพันธ์กันสูงมาก (ราคาเคลื่อนไหวไปด้วยกัน)
# ส่วน Volume มักมีความสัมพันธ์กับราคาน้อยกว่าอย่างชัดเจน

# ============================================================
# PART 3: Data Cleaning
# ============================================================

# %% Part3.1: Incorrect Data Correction
# Volume = 0 ในวันที่มีการซื้อขาย ถือเป็นค่าผิดปกติ (ไม่สมเหตุสมผลสำหรับหุ้นขนาดใหญ่อย่าง MSFT)
n_zero_volume = (df["Volume"] == 0).sum()
print("จำนวนแถวที่ Volume = 0 (ผิดปกติ):", n_zero_volume)
df.loc[df["Volume"] == 0, "Volume"] = np.nan

# ราคา (Open/High/Low/Close) ที่ติดลบหรือเป็น 0 ก็ผิดปกติเช่นกัน
price_cols = ["Open", "High", "Low", "Close"]
for col in price_cols:
    n_bad = (df[col] <= 0).sum()
    if n_bad > 0:
        print(f"พบค่าผิดปกติ (<=0) ใน {col}: {n_bad} แถว")
        df.loc[df[col] <= 0, col] = np.nan

# %% Part3.2: Data Type Conversion
# แปลง Date จาก string -> datetime เพื่อให้เรียงลำดับเวลาและดึงฟีเจอร์ย่อยได้ถูกต้อง
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
print(df["Date"].dtype)
print("ช่วงข้อมูล:", df["Date"].min(), "ถึง", df["Date"].max())

# %% Part3.3: Compare Mean vs Median (ของ Volume ก่อนเติมค่า)
mean_vol = df["Volume"].mean()
median_vol = df["Volume"].median()
print(f"Mean Volume   : {mean_vol:,.0f}")
print(f"Median Volume : {median_vol:,.0f}")

plt.figure(figsize=(6, 4))
sns.histplot(df["Volume"].dropna(), kde=True)
plt.axvline(mean_vol, color="red", linestyle="--", label=f"Mean = {mean_vol:,.0f}")
plt.axvline(median_vol, color="green", linestyle="--", label=f"Median = {median_vol:,.0f}")
plt.legend()
plt.title("Volume: Mean vs Median")
plt.tight_layout()
plt.show()
# Volume มักมีการกระจายตัวแบบเบ้ขวารุนแรง (มีบางวันซื้อขายหนักผิดปกติ เช่น วันประกาศผลประกอบการ)
# ทำให้ Mean ถูกดึงสูงกว่าค่ากลางจริงมาก -> เลือกใช้ Median ในการเติมค่าจะเหมาะสมกว่า

# %% Part3.4: Missing Value Handling
print("ค่าที่หายไปก่อนเติม:")
print(df[price_cols + ["Volume"]].isnull().sum())

# ราคา (Open/High/Low/Close) เป็น time series ต่อเนื่อง -> ใช้ forward fill
# (เอาค่าราคาปิดของวันก่อนหน้ามาแทน เหมาะกับข้อมูลการเงินมากกว่าค่าเฉลี่ย/มัธยฐานทั้งชุด)
df[price_cols] = df[price_cols].ffill()

# Volume ใช้ median เพราะข้อมูลเบ้และไม่ต่อเนื่องเชิงแนวโน้มแบบราคา
df["Volume"] = df["Volume"].fillna(median_vol)

print("ค่าที่หายไปหลังเติม:", df[price_cols + ["Volume"]].isnull().sum().sum())

# %% Part3.5: Duplicate Removal
print("จำนวนแถวซ้ำก่อนลบ:", df.duplicated().sum())
df = df.drop_duplicates(subset="Date", keep="first")
print("จำนวนแถวซ้ำหลังลบ (ตาม Date):", df.duplicated(subset="Date").sum())
print("Shape หลังทำความสะอาดข้อมูล:", df.shape)

# ============================================================
# PART 4: Feature Engineering
# ============================================================

# %% Part4.1: Label Encoding (สร้าง Trend แบบ binary แล้ว encode)
df["Trend"] = np.where(df["Close"].diff() > 0, "Up", "Down")
df.loc[df.index[0], "Trend"] = "Down"  # แถวแรกไม่มีวันก่อนหน้าให้เทียบ กำหนดค่าเริ่มต้นไว้

le = LabelEncoder()
df["Trend"] = le.fit_transform(df["Trend"])  # Up/Down -> 1/0

print("ตัวอย่างข้อมูลหลัง Label Encoding:")
print(df[["Date", "Close", "Trend"]].head())
# หมายเหตุ: Trend มีแค่ 2 ค่า (Up/Down) จึงเหมาะกับ Label Encoding
# ไม่ทำให้เกิดความสัมพันธ์เชิงลำดับที่ไม่มีอยู่จริง

# %% Part4.2: One-Hot Encoding (สร้าง Quarter จากวันที่ แล้ว encode)
df["Quarter"] = df["Date"].dt.quarter.map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"})
df = pd.get_dummies(df, columns=["Quarter"], drop_first=True)

print("Shape หลังทำ One-Hot Encoding:", df.shape)
df.head()
# หมายเหตุ: Quarter มี 4 ค่าไม่มีลำดับความสำคัญเชิงตัวเลข จึงใช้ One-Hot Encoding
# แทนการปล่อยเป็นตัวเลข 1-4 ที่จะทำให้โมเดลเข้าใจผิดว่ามีลำดับ

# %% [Summary] สรุปผลลัพธ์สุดท้าย
print("=" * 50)
print("สรุปข้อมูลหลัง Preprocessing ทั้งหมด")
print("=" * 50)
print("Shape สุดท้าย:", df.shape)
print("จำนวน missing values คงเหลือ:", df.isnull().sum().sum())
print("จำนวนแถวซ้ำคงเหลือ:", df.duplicated().sum())
print(df.dtypes.value_counts())

# (ถ้าต้องการบันทึกไฟล์ที่ทำความสะอาดแล้ว)
# df.to_csv("msft_cleaned.csv", index=False)
