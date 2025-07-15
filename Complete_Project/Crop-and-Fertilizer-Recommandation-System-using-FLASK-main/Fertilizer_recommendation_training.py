import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the dataset
df = pd.read_csv('Fertilizer.csv')

# Encode the target variable (Fertilizer Name)
label_encoder = LabelEncoder()
df['Fertilizer Name'] = label_encoder.fit_transform(df['Fertilizer Name'])

# Separate features and target
X = df[['Nitrogen', 'Potassium', 'Phosphorous']]
y = df['Fertilizer Name']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to train and evaluate models
def train_and_evaluate_model(model, model_name):
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print accuracy and classification report
    print(f"\n{model_name} Accuracy: {accuracy:.3f}")
    print(f"{model_name} Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42)
train_and_evaluate_model(log_reg, "Logistic Regression")

# 2. Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(random_state=42)
train_and_evaluate_model(decision_tree, "Decision Tree")

# 3. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
train_and_evaluate_model(random_forest, "Random Forest")

# 4. K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
train_and_evaluate_model(knn, "K-Nearest Neighbors")

# 5. Support Vector Machine (SVM)
from sklearn.svm import SVC
svm = SVC(probability=True, random_state=42)
train_and_evaluate_model(svm, "Support Vector Machine")

# 6. Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
train_and_evaluate_model(naive_bayes, "Naive Bayes")

# Visualize the Accuracy of all models
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN', 'SVM', 'Naive Bayes']
accuracies = [
    accuracy_score(y_test, log_reg.predict(X_test_scaled)),
    accuracy_score(y_test, decision_tree.predict(X_test_scaled)),
    accuracy_score(y_test, random_forest.predict(X_test_scaled)),
    accuracy_score(y_test, knn.predict(X_test_scaled)),
    accuracy_score(y_test, svm.predict(X_test_scaled)),
    accuracy_score(y_test, naive_bayes.predict(X_test_scaled))
]

# Plotting accuracies of different models
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple', 'orange', 'cyan'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Save the best model (assuming Random Forest performed the best)
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(random_forest, file)
print("Random Forest model saved to random_forest_model.pkl")

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
print("Scaler saved to scaler.pkl")

# Load the saved model and make a prediction
with open('random_forest_model.pkl', 'rb') as file:
    loaded_rf_model = pickle.load(file)

# Example input data for prediction (make sure the dimensions match your model)
sample_data = np.array([[110, 82, 91]])  # Example: N, P, K values for fertilizer prediction
sample_data_scaled = scaler.transform(sample_data)

# Make a prediction
prediction = loaded_rf_model.predict(sample_data_scaled)
predicted_fertilizer = label_encoder.inverse_transform(prediction)
print("Predicted Fertilizer Name:", predicted_fertilizer[0])