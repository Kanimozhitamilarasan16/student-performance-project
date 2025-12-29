import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

if not os.path.exists("outputs"):
    os.makedirs("outputs")

data = pd.read_csv("datasets/student-mat.csv", sep=';', quotechar='"')

categorical_columns = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian',
                       'schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']

encoder = LabelEncoder()
for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col])

grade_encoder = LabelEncoder()
data['G3'] = grade_encoder.fit_transform(data['G3'])

X = data.drop('G3', axis=1)
y = data['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

number_of_classes = len(y.unique())

model = XGBClassifier(objective='multi:softmax',
                      num_class=number_of_classes,
                      n_estimators=300,
                      max_depth=5,
                      learning_rate=0.1,
                      eval_metric='mlogloss',
                      use_label_encoder=False,
                      random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

with open("outputs/classification_report.txt", "w") as f:
    f.write("Accuracy: " + str(acc) + "\n\n")
    f.write(report)

plt.figure(figsize=(8,5))
sns.countplot(x='G3', data=data, palette='coolwarm')
plt.title("Grade Distribution (G3)")
plt.savefig("outputs/grade_distribution.png")
plt.close()

plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

print("All outputs are saved in the 'outputs' folder.")


