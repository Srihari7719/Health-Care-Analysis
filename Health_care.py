# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# %%
df = pd.read_csv('healthcare_dataset.csv')
df.head()

# %%
df = df[df['Test Results'] != 'Inconclusive']


# %%
df['Test Results'].value_counts()

# %%
df.shape

# %%
df.isna().sum()

# %% [markdown]
# # **EDA**

# %%
# Pairplot for numeric features
numeric_features = ['Age', 'Billing Amount']
sns.pairplot(df[numeric_features])
plt.show()

# %%
# Distribution of Age
plt.figure(figsize=(10, 5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Age')
plt.show()

# %%
# Boxplot of Billing Amount by Admission Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Admission Type', y='Billing Amount', data=df)
plt.xlabel('Admission Type')
plt.ylabel('Billing Amount')
plt.title('Billing Amount by Admission Type')
plt.show()

# %% [markdown]
# # Feature Engineering 

# %%
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
df[[ 'Length of Stay']]

# %%
# feature Engineering 
# not needed columns - [Name, Date, Doctor, Hospital, amount, room, discharge]
data = df[[ 'Age', 'Gender','Blood Type', 'Medical Condition','Admission Type','Insurance Provider', 'Medication', 'Test Results']]
df.head()

# %%
# Define age group intervals and labels
bins = [0, 30, 50, 70, 100]  # Age group intervals
labels = ['Under 30', '31-50', '51-70', 'Over 70']  # Labels for age groups

# Add a new column 'Age Group' based on age intervals
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

df[['Age', 'Age Group']]

# %% [markdown]
# ### Number of uniques values

# %%
for col in df.columns:
    print(col, df[col].nunique())

# %%
average_billing_per_doctor = df.groupby('Doctor')['Billing Amount'].mean()
print(average_billing_per_doctor)

# %% [markdown]
# ### Label encoding

# %%
age_bins = [0, 20, 40, 60, 80, float('inf')]
age_labels = ['0-20', '21-40', '41-60', '61-80', '81+']
df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
df

# %%
variables_to_remove = ['Name','Date of Admission','Hospital','Room Number','Discharge Date','Doctor','Age']

df = df.drop(variables_to_remove, axis=1)
df

# %%
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col]=lc.fit_transform(df[col])

df['Age Group'] = lc.fit_transform(df['Age Group'])
df.head()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = df.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# %%
df

# %% [markdown]
# Model fitting

# %%
X,y=df.drop(['Test Results'],axis=1), df['Test Results']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# %%
X_train.shape, y_test.shape

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# %%
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svc_model = SVC()

svc_model.fit(X_train, y_train)

y_pred = svc_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)


# %%

y = df['Test Results']  
X = df.drop(columns=['Test Results']) 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


logistic_regression = LogisticRegression()

logistic_regression.fit(X_train, y_train)

y_pred = logistic_regression.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)


# %%
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))



