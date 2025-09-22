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
```

### Handling Missing Values

- **Drop**: `Cabin` (too many missing values)  
- **Median Imputation**: `Age` (robust to outliers)  
- **Mode Imputation**: `Embarked` (categorical)  

```python
df = df.drop(columns=['Cabin'])
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
```

### 3. Encode Categorical Features

- Label Encoding: Sex → 0/1.
- One-Hot Encoding: `Embarked` → dummy variables.
```python
from sklearn.preprocessing import LabelEncoder
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
```

### 4. Normalize / Standardize Numerical Features
- Standardize `Age`, `Fare`, `SibSp`, `Parch` using StandardScaler.
```python
from sklearn.preprocessing import StandardScaler
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
```

### 5. Outlier Detection & Removal
- Visualize outliers with boxplots.
- Remove outliers using the IQR method.
```python
  plt.figure(figsize=(12,6))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2,2,i)
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

def remove_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    return data[(data[col] >= lower) & (data[col] <= upper)]

for col in num_cols:
    df = remove_outliers_iqr(df, col)

print("Shape after outlier removal:", df.shape)
```

### Final Dataset
- Cleaned, imputed, encoded, scaled, and outlier-free.

