import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

sns.set_style("whitegrid")

df=pd.read_csv('fraud_detection_dataset_10000.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()

# Remove duplicates
df.drop_duplicates(subset='Transaction_ID', inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df.isnull().sum()

df['Status'].value_counts()
sns.countplot(x='Status', data=df)
plt.title("Fraud vs Normal Transactions")
plt.show()

plt.hist(df["Amount"], bins=40)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()

sns.countplot(x="Payment_Method", hue="Status", data=df)
plt.xticks(rotation=45)
plt.title("Fraud by Payment Method")
plt.show()

fraud_month = df[df["Status"]=="Suspicious"].groupby("Month").size()

fraud_month.plot(kind="bar")
plt.title("Monthly Suspicious Transactions")
plt.show()


iso = IsolationForest(contamination=0.05)

df['Anomaly_Score'] = iso.fit_predict(df[['Amount']])

# Convert -1 to Suspicious, 1 to Normal
df['Anomaly_Label'] = df['Anomaly_Score'].map({1:'Normal', -1:'Suspicious'})

df[['Amount','Anomaly_Label']].head()

df.to_csv("cleaned_fraud_data.csv")

df['Risk_Flag'] = np.where(df['Status']=='Suspicious',1,0)

# Encode Payment Method
df = pd.get_dummies(df, columns=['Payment_Method'], drop_first=True)
X = df[['Amount','Month'] + [col for col in df.columns if 'Payment_Method_' in col]]
y = df['Risk_Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Simulate next month
next_month = df['Month'].max() + 1

future_data = X.copy()
future_data['Month'] = next_month

future_prediction = model.predict(future_data)

df['Next_Month_Risk_Prediction'] = future_prediction

df[['Transaction_ID','Next_Month_Risk_Prediction']].head()

df.to_csv("fraud_analysis_final.csv")