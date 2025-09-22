# 🛳️ Titanic Dataset Preprocessing Pipeline

This project demonstrates a complete **data preprocessing workflow** on the famous [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic).  
The goal is to **clean, transform, and prepare the data** for machine learning models.

---

## 📂 Steps in the Pipeline

### 1. Import Dataset & Explore
Load the dataset and inspect its structure.

```python
import pandas as pd

df = pd.read_csv("train.csv")
print(df.info())
print(df.isnull().sum())


