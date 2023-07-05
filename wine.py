import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

filename = "wine_combined_cleaned.csv"  # Replace with the actual file name


# Read the CSV file into a DataFrame
df = pd.read_csv(filename)

# Print the DataFrame

# print(df.head())

X = df.iloc[:,:12].values
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a Decision Tree Classifier
classifier = DecisionTreeClassifier()

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, predictions)
print("DT Accuracy:", accuracy)

# Read the CSV file into a DataFrame
df = pd.read_csv(filename)

# Print the DataFrame

# print(df.head())

X = df.iloc[:,:12].values
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a Decision Tree Classifier
classifier = RandomForestClassifier()

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, predictions)

print("RF Accuracy:", accuracy)