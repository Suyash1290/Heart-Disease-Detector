import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

file_path = r"D:\Round 1\Heart disease detector\heart_disease_data.csv"
data = pd.read_csv(file_path)

X = data.drop('target', axis=1)
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model.fit(X_scaled, y)

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = scaler.transform([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        return "You are likely to have heart disease."
    else:
        return "You are unlikely to have heart disease."

age = int(input("Enter your age: "))
sex = int(input("Enter your sex (0 for female, 1 for male): "))
cp = int(input("Enter your chest pain type (0-3): "))
trestbps = int(input("Enter your resting blood pressure (in mm Hg): "))
chol = int(input("Enter your serum cholesterol level (in mg/dl): "))
fbs = int(input("Enter your fasting blood sugar (> 120 mg/dl) (0 for No, 1 for Yes): "))
restecg = int(input("Enter your resting electrocardiographic results (0-2): "))
thalach = int(input("Enter your maximum heart rate achieved: "))
exang = int(input("Enter whether you have exercise induced angina (0 for No, 1 for Yes): "))
oldpeak = float(input("Enter your ST depression induced by exercise relative to rest: "))
slope = int(input("Enter the slope of the peak exercise ST segment (0-2): "))
ca = int(input("Enter the number of major vessels colored by fluoroscopy (0-3): "))
thal = int(input("Enter your thalassemia (0-3): "))

result = predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
print(result)
