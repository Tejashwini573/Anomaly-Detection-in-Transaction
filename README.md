# Anomaly-Detection-in-Transaction
Overview

This project focuses on detecting fraudulent financial transactions using machine learning techniques. The goal is to identify unusual patterns in transaction data and classify them as normal or suspicious.

The project includes data preprocessing, exploratory data analysis (EDA), anomaly detection, classification modeling, and dashboard visualization to deliver end-to-end insights.

🎯 Objectives
Detect fraudulent transactions using machine learning models
Perform data cleaning and preprocessing
Analyze transaction patterns through EDA
Build predictive models for fraud detection
Visualize insights using Power BI dashboard
📂 Dataset
Contains financial transaction records
Includes features like:
Transaction ID
Date
Transaction Amount
Status (Fraud / Legitimate)
🛠️ Technologies Used
Python
Pandas, NumPy → Data manipulation
Matplotlib, Seaborn → Data visualization
Scikit-learn → Machine Learning models
Power BI → Dashboard creation
🔍 Data Preprocessing
Removed duplicate records using Transaction ID
Converted date column to datetime format
Extracted Month and Year from date
Checked and handled missing values
📊 Exploratory Data Analysis (EDA)

EDA was performed to understand patterns and trends:

Distribution of transaction status (Fraud vs Non-Fraud)
Transaction trends over time
Visualization using count plots and graphs
Identified imbalance in fraud detection
🤖 Machine Learning Models
1. Isolation Forest
Used for anomaly detection
Identifies outliers in transaction data
Works well for unsupervised fraud detection
2. Logistic Regression
Used for classification
Predicts whether a transaction is fraud or not
📈 Model Evaluation

The model performance was evaluated using:

Accuracy Score
Confusion Matrix
Classification Report (Precision, Recall, F1-score)

These metrics help in understanding how well the model identifies fraudulent transactions.

📊 Dashboard

An interactive Power BI dashboard was created to:

Monitor fraud trends
Analyze transaction behavior
Visualize key metrics like fraud rate and transaction distribution
📁 Project Structure
├── anomaly_detections.py
├── fraud_detection_dataset_10000.csv
├── fraud_analysis_final.csv
├── Anomaly detection dashboard.pbix
├── README.md
🚀 How to Run the Project
Clone the repository
git clone https://github.com/your-username/your-repo-name.git
Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn
Run the Python script
python anomaly_detections.py
💡 Key Insights
Fraudulent transactions are relatively rare (imbalanced dataset)
Certain patterns in transaction behavior indicate anomalies
Machine learning models can effectively detect suspicious activity
🔮 Future Improvements
Implement advanced models (Random Forest, XGBoost)
Handle class imbalance using SMOTE
Deploy model using Flask or Streamlit
Real-time fraud detection system
🙌 Conclusion

This project demonstrates how machine learning can be applied to detect financial fraud effectively. It combines data analysis, modeling, and visualization to provide meaningful insights into transaction anomalies.
