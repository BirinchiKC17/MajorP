import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # or LogisticRegression, etc.
from sklearn.metrics import accuracy_score

columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv('pima-indians-diabetes.data.csv', names=columns)

print(df.head())

# Example columns: 'Glucose', 'BMI', 'Outcome'
X = df[['Glucose', 'BMI']]   # Features
y = df['Outcome']            # Target (1 = diabetic, 0 = non-diabetic)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Example input (e.g., from input() or another source)
glucose = float(input("Enter glucose level: "))
bmi = float(input("Enter BMI: "))

# Must reshape input to match training shape
input_data = ['Glucose', 'BMI']
new_data =[[glucose,bmi]]
df_new_data = pd.DataFrame(new_data, columns=['Glucose','BMI'])
prediction = model.predict(df_new_data)
probability = model.predict_proba(df_new_data)

print(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")
print(f"Probability of being diabetic: {probability[0][1]:.2f}")
