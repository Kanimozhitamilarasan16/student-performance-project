Student Performance Project

This project predicts the final grades (G3) of students based on various factors such as demographic information, family background, and academic performance. It is built as part of the AIML coursework.

Project Overview:

The dataset contains **395 student records** with **33 features**, including:

- **Demographics:** age, sex, address, family size, parents’ status
- **Academic background:** past grades (G1, G2), study time, failures
- **Family & social factors:** parental education, family support, activities
- **Health & lifestyle:** alcohol consumption, free time, health, romantic relationships
- **Internet access & nursery attendance**

The goal is to **predict the final grade (G3)** using machine learning models.

Project Files:

- 'student_ml.py' – Main Python code implementing the ML model
- 'datasets/student-mat.csv' – Dataset used for training and testing
- 'outputs/' – Folder for generated charts, predictions, and results

Usage:

1. Install required Python libraries:

   pip install pandas scikit-learn xgboost matplotlib seaborn

2. Run the project:

   python student_ml.py

3. Check the 'outputs/' folder for generated graphs and predictions.

Model & Method:

- Preprocessing: Encoding categorical variables
- Model: XGBoost Classifier for predicting final grades
- Evaluation: Accuracy, classification report, confusion matrix
- Visualization: Histograms, distribution plots for analysis

