import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#"D:\project_sem7\cardio_train.csv"

# Load your dataset
data = pd.read_csv('D:\project_sem7\cardio_train.csv', sep=';')  # Ensure correct path and separator

# Adjust here
features = data.drop('cardio', axis=1)  # Drop the target variable
labels = data['cardio']  # Use 'cardio' as the label

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
