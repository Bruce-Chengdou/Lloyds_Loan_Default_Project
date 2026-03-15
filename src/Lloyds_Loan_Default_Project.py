#!/usr/bin/env python
# coding: utf-8

# BRUCE CHENGDOU WEI

# Lloyds Banking Group Project

# In[1]:


# 1 Project Background
# 2 Data Cleaning
# 3 Feature Engineering
# 4 Exploratory Analysis
# 5 Model Development
# 6 Model Insights


# In[2]:


import pandas as pd
from pathlib import Path

data_path = Path("../data/loans_dataset.xlsx")

df = pd.read_excel(data_path)

print(df.shape)
df.head()


# In[3]:


# 1 Project Background


# In[4]:


# check loan status
df["loan_status"].value_counts()


# In[5]:


# percentage 
df["loan_status"].value_counts(normalize=True)


# In[6]:


# type
df.dtypes


# In[7]:


# what is lost 1
df.isna().sum().sort_values(ascending=False)


# In[8]:


# what is lost 2
(df.isna().mean()*100).sort_values(ascending=False)


# In[9]:


df["emp_length"].unique()


# In[10]:


# Data Cleaning


# In[11]:


# transfer emp_length into float
df["emp_length"] = df["emp_length"].replace({
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10
})

df["emp_length"] = df["emp_length"].astype(float)

df["emp_length"].unique()


# In[12]:


# 3 Feature Engineering


# In[13]:


# delete ID
df = df.drop(columns=["id"])

# transfer loan_status into int
df["loan_status"] = df["loan_status"].map({
    "Fully Paid": 0,
    "Charged Off": 1
}).astype(int)

# transfer term into int
df["term"] = df["term"].str.replace(" months","").astype(int)

df.head()


# In[14]:


# check types of the values
for col in df.select_dtypes(include="object").columns:
    print(col, df[col].nunique())


# In[15]:


# emp_title is a mess
df = df.drop(columns=["emp_title"])
df.shape


# In[16]:


# for the missing values

import numpy as np

# find all the num
numeric_cols = df.select_dtypes(include=[np.number]).columns

# median imputation
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# check
df.isna().sum().sort_values(ascending=False)


# In[17]:


# get_dummies() for One-Hot Encoding
# drop_first=True for avoiding dummy variable trap

categorical_cols = ["addr_state", "home_ownership", "purpose"]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# check
df.shape

df.head()


# In[18]:


# save figure
from pathlib import Path


# In[19]:


# check distribution shape
# could be fun

import matplotlib.pyplot as plt
import seaborn as sns

# Annual income
plt.figure(figsize=(6,4))
sns.histplot(df["annual_inc"], bins=50)
plt.title("Distribution of Annual Income")

plt.savefig(Path("../outputs/annual_income_distribution.png"),
            dpi=300,
            bbox_inches="tight")

plt.show()


# In[20]:


# check distribution shape 2
# loan amount

plt.figure(figsize=(6,4))
sns.histplot(df["loan_amnt"], bins=50)
plt.title("Distribution of Loan Amount")

plt.savefig(Path("../outputs/Loan_Amount_distribution.png"),
            dpi=300,
            bbox_inches="tight")

plt.show()


# In[21]:


# check distribution shape 3
# interest rate

plt.figure(figsize=(6,4))
sns.histplot(df["int_rate"], bins=50)
plt.title("Distribution of Interest Rate")

plt.savefig(Path("../outputs/Interest_Rate_distribution.png"),
            dpi=300,
            bbox_inches="tight")

plt.show()


# In[22]:


# check distribution shape 4
# Total Balance Excluding Mortgage

plt.figure(figsize=(6,4))
sns.histplot(df["total_bal_ex_mort"], bins=50)
plt.title("Distribution of Total Balance Excluding Mortgage")

plt.savefig(Path("../outputs/Total_Balance_Excluding_Mortgage_distribution.png"),
            dpi=300,
            bbox_inches="tight")

plt.show()


# In[23]:


# check distribution shape 5
# Average Current Balance

plt.figure(figsize=(6,4))
sns.histplot(df["avg_cur_bal"], bins=50)
plt.title("Distribution of Average Current Balance")

plt.savefig(Path("../outputs/Average_Current_Balance_distribution.png"),
            dpi=300,
            bbox_inches="tight")

plt.show()


# In[24]:


# 4 Exploratory Analysis


# In[25]:


# Risk Segmentation

df.groupby("loan_status")[[
    "annual_inc",
    "loan_amnt",
    "int_rate",
    "installment",
    "avg_cur_bal",
    "total_bal_ex_mort",
    "inq_last_12m",
    "percent_bc_gt_75"
]].mean()


# In[26]:


# 5 Model Development


# In[27]:


X = df.drop("loan_status", axis=1)
y = df["loan_status"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)


# In[28]:


# Logistic Regression baseline

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


# In[29]:


# accuracy
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


# In[30]:


# classification report

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[31]:


# the recall is 0, unacceptable 


# In[32]:


# roc and auc
from sklearn.metrics import roc_auc_score

y_prob = model.predict_proba(X_test)[:,1]

roc_auc_score(y_test, y_prob)


# In[33]:


# adjust class weight

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# In[34]:


# Feature Importance


# In[35]:


# Coefficients
import pandas as pd

coef = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0]
})

coef = coef.sort_values(by="coefficient", ascending=False)

coef.head(15)


# In[36]:


coef.tail(15)


# In[37]:


# threshold tuning
# it was P(default) > 0.5, we can lower this, so can catch more. we use 0.3 here.

y_prob = model.predict_proba(X_test_scaled)[:,1]

import numpy as np

y_pred_03 = (y_prob > 0.3).astype(int)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_03))


# In[38]:


# to avoid too many False Positives, keep the default threshold 0.5


# In[39]:


# 6 Model Insights


# In[40]:


# ROC curve

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,4))
# plt.plot(fpr, tpr, label="Logistic Regression (AUC = 0.69)")
auc = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.2f})")
plt.plot([0,1],[0,1],'--', color="grey")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Loan Default Model")
plt.legend()

plt.savefig(Path("../outputs/ROC_Curve.png"),
            dpi=300,
            bbox_inches="tight")

plt.show()


# In[41]:


# Top 15 Risk Factors

coef = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0]
})

coef = coef.sort_values(by="coefficient", ascending=False)

top_risk = coef.head(15)
top_risk

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

plt.barh(top_risk["feature"], top_risk["coefficient"])

plt.xlabel("Coefficient Value")
plt.title("Top 15 Risk Factors for Loan Default")

plt.gca().invert_yaxis()

plt.savefig(Path("../outputs/TOP15_Risk_Factors.png"),
            dpi=300,
            bbox_inches="tight")

plt.show()


# In[42]:


# feature label mapping

feature_labels = {
    "int_rate": "Interest Rate",
    "term": "Loan Term",
    "installment": "Monthly Installment",
    "inq_last_12m": "Credit Inquiries (12m)",
    "num_tl_op_past_12m": "Accounts Opened (12m)",
    "num_op_rev_tl": "Open Revolving Accounts",
    "num_il_tl": "Installment Accounts",
    "pub_rec_bankruptcies": "Bankruptcy Records",
    "home_ownership_RENT": "Home Ownership: Rent",
    "home_ownership_MORTGAGE": "Home Ownership: Mortgage",
    "home_ownership_OWN": "Home Ownership: Own",
    "purpose_small_business": "Loan Purpose: Small Business",
    "purpose_debt_consolidation": "Loan Purpose: Debt Consolidation",
    "purpose_moving": "Loan Purpose: Moving",
    "purpose_educational": "Loan Purpose: Education"
}

# Apply label mapping

top_risk_plot = top_risk.copy()

top_risk_plot["label"] = top_risk_plot["feature"].map(feature_labels)

# the rest keep the same
top_risk_plot["label"] = top_risk_plot["label"].fillna(top_risk_plot["feature"])


# In[43]:


# re plot

plt.figure(figsize=(8,6))

plt.barh(top_risk_plot["label"], top_risk_plot["coefficient"])

plt.xlabel("Coefficient Value")
plt.title("Top 15 Risk Factors for Loan Default")

plt.gca().invert_yaxis()

plt.savefig(
    Path("../outputs/Key_Risk_Factors_of_Loan_Default.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[ ]:





# In[ ]:




