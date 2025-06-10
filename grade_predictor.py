import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print("Script is running...")


# Step 1: Load Data
data = pd.read_csv("student_scores.csv")

# Step 2: Features and Label
X = data[['Previous_Test1', 'Previous_Test2']]
y = data['Final_Grade']

# Step 3: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict
y_pred = model.predict(X_test)

# Step 6: Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Optional: Predict new data
new_data = [[85, 90]]  # Example scores of a student
predicted_grade = model.predict(new_data)
print(f"Predicted Final Grade: {predicted_grade[0]:.2f}")
