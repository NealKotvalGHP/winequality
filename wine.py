import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

filename = "wine_combined_cleaned.csv"  # Replace with the actual file name

# Load the data into a DataFrame
df = pd.read_csv(filename)

# Split the data into features (X) and target variable (y)
X = df.iloc[:, :-1]  # First 12 columns as features
y = df.iloc[:, -1]   # Last column as target variable

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create individual classifiers
random_forest = RandomForestClassifier()
decision_tree = DecisionTreeClassifier()

# Train individual classifiers
random_forest.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)

# Make predictions on the test set
rf_pred = random_forest.predict(X_test)
dt_pred = decision_tree.predict(X_test)

# Calculate accuracies of individual classifiers
rf_accuracy = accuracy_score(y_test, rf_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)

# Calculate weights based on accuracies
rf_weight = rf_accuracy / (rf_accuracy + dt_accuracy)
dt_weight = dt_accuracy / (rf_accuracy + dt_accuracy)

print(rf_pred)
print(dt_pred)
# Create the weighted ensemble using majority voting
ensemble1 = VotingClassifier(estimators=[('rf', random_forest), ('dt', decision_tree)], voting='hard', weights=[rf_weight, dt_weight])

# Train the ensemble
ensemble1.fit(X_train, y_train)

# Make predictions on the test set
e1_pred = ensemble1.predict(X_test)

# Create the weighted ensemble using majority voting
ensemble2 = VotingClassifier(estimators=[('rf', random_forest), ('dt', decision_tree)], voting='hard', weights=[rf_weight, dt_weight])

# Train the ensemble
ensemble2.fit(X_train, y_train)

# Make predictions on the test set
e2_pred = ensemble2.predict(X_test)

e1_accuracy = accuracy_score(y_test, e1_pred)
e2_accuracy = accuracy_score(y_test, e2_pred)


# Calculate weights based on accuracies
e1_weight = e1_accuracy / (e1_accuracy + e2_accuracy)
e2_weight = e2_accuracy / (e1_accuracy + e2_accuracy)



ensembleEnsemble = VotingClassifier(estimators=[('e1', ensemble1), ('e2', ensemble2)], voting='hard', weights=[e1_weight, e2_weight])

ensembleEnsemble.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ensembleEnsemble.predict(X_test)

# Evaluate the accuracy of the ensemble
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)