import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("dataset/student_data.csv")

# Features
X = data[["Attendance", "CGPA", "Backlogs", "StudyHours"]]

# Target
y = data["Dropout"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, prediction)

print(f"Model Accuracy : {accuracy*100:.2f}%")

# Save Model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/student_dropout_model.pkl")

print("Model Saved Successfully.")