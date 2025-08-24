#Autism Spectrum Disorder Detection using Machine Learning

📌 Project Overview

This project explores the use of machine learning models to detect Autism Spectrum Disorder (ASD) based on behavioral and screening data. The primary goal was to build predictive models that can assist in early diagnosis, thereby supporting clinicians and caregivers in timely intervention.

🛠️ Work Done

Conducted data preprocessing including handling missing values, feature encoding, and normalization.

Performed exploratory data analysis (EDA) to identify correlations and feature importance.

Implemented multiple machine learning algorithms for classification.

Tuned hyperparameters and compared models on accuracy, precision, recall, and F1-score.

Visualized results through confusion matrices and performance plots.

📊 Dataset Used

Source: Autism Screening Dataset (children and adults) from UCI Machine Learning Repository.

Features:

Age, gender, ethnicity

Screening responses (Yes/No based on behavioral traits)

Family history of ASD

Screening test scores

Target Variable: ASD diagnosis (Yes/No).

🤖 Machine Learning Models Implemented

Logistic Regression

Decision Trees

Random Forests

Support Vector Machines (SVM)

K-Nearest Neighbors (KNN)

Naïve Bayes

📈 Results

Models achieved strong performance, with Random Forest and SVM performing the best.

Random Forest achieved the highest accuracy (~96%) with balanced precision and recall.

Feature analysis highlighted screening responses and family history as the most important predictors.

The models showed potential for integration into clinical decision support systems.

🚀 Technical Details

Language/Frameworks: Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

Workflow:

Data preprocessing & cleaning

Model training and evaluation

Hyperparameter tuning

Performance comparison

Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix


This project demonstrates that machine learning models, particularly ensemble methods like Random Forests, can effectively detect Autism Spectrum Disorder using behavioral screening data. While the results are promising, real-world deployment would require larger and more diverse datasets, along with validation by healthcare professionals.
