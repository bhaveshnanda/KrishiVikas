# import os
# import pandas as pd
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# # Load the dataset
# crop = pd.read_csv('Crop_recommendation.csv')

# # Preprocessing
# crop_dict = { 
#     'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
#     'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
#     'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 
#     'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
#     'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
# }
# crop['crop_num'] = crop['label'].map(crop_dict)
# crop = crop.drop('label', axis=1)

# # Features and target variable
# X = crop.drop('crop_num', axis=1) 
# Y = crop['crop_num']

# # Split the data into training and testing sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # Standardize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Define models
# log_reg = LogisticRegression()
# decision_tree = DecisionTreeClassifier()
# random_forest = RandomForestClassifier()
# knn = KNeighborsClassifier()
# svm = SVC()
# naive_bayes = GaussianNB()

# # Train models
# log_reg.fit(X_train, Y_train)
# decision_tree.fit(X_train, Y_train)
# random_forest.fit(X_train, Y_train)
# knn.fit(X_train, Y_train)
# svm.fit(X_train, Y_train)
# naive_bayes.fit(X_train, Y_train)

# # Predictions
# y_pred_log_reg = log_reg.predict(X_test)
# y_pred_decision_tree = decision_tree.predict(X_test)
# y_pred_random_forest = random_forest.predict(X_test)
# y_pred_knn = knn.predict(X_test)
# y_pred_svm = svm.predict(X_test)
# y_pred_naive_bayes = naive_bayes.predict(X_test)

# # Calculate accuracy
# accuracies = {
#     'Logistic Regression': accuracy_score(Y_test, y_pred_log_reg),
#     'Decision Tree': accuracy_score(Y_test, y_pred_decision_tree),
#     'Random Forest': accuracy_score(Y_test, y_pred_random_forest),
#     'K-Nearest Neighbors': accuracy_score(Y_test, y_pred_knn),
#     'Support Vector Machine': accuracy_score(Y_test, y_pred_svm),
#     'Naive Bayes': accuracy_score(Y_test, y_pred_naive_bayes)
# }

# # Print accuracies
# for model_name, accuracy in accuracies.items():
#     print(f"{model_name} Accuracy: {accuracy:.4f}")

# # Save the Naive Bayes model
# with open('naive_bayes_model.pkl', 'wb') as f:
#     pickle.dump(naive_bayes, f)

# # Function to display confusion matrix for each model
# def plot_confusion_matrix(y_true, y_pred, model_name):
#     cm = confusion_matrix(y_true, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     disp.plot(cmap=plt.cm.Blues)
#     plt.title(f'Confusion Matrix for {model_name}')
#     plt.show()

# # Display confusion matrices for all models
# plot_confusion_matrix(Y_test, y_pred_log_reg, 'Logistic Regression')
# plot_confusion_matrix(Y_test, y_pred_decision_tree, 'Decision Tree')
# plot_confusion_matrix(Y_test, y_pred_random_forest, 'Random Forest')
# plot_confusion_matrix(Y_test, y_pred_knn, 'K-Nearest Neighbors')
# plot_confusion_matrix(Y_test, y_pred_svm, 'Support Vector Machine')
# plot_confusion_matrix(Y_test, y_pred_naive_bayes, 'Naive Bayes')

# # Accuracy Comparison Chart
# plt.figure(figsize=(10, 6))
# plt.bar(accuracies.keys(), accuracies.values(), color='royalblue')
# plt.xlabel('Models')
# plt.ylabel('Accuracy')
# plt.title('Model Accuracy Comparison')
# plt.ylim(0, 1)
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

# Importing of libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Crop dictionary mapping numbers to names
crop_dict = { 
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut', 6: 'papaya',
    7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon', 11: 'grapes', 12: 'mango',
    13: 'banana', 14: 'pomegranate', 15: 'lentil', 16: 'blackgram', 17: 'mungbean',
    18: 'mothbeans', 19: 'pigeonpeas', 20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
}
# Reverse dictionary
crop_name_to_number = {v: k for k, v in crop_dict.items()}

# Load dataset
df_crop = pd.read_csv('Crop_recommendation.csv')

# Replace crop names with corresponding numbers
df_crop['label'] = df_crop['label'].map(crop_name_to_number)

# Distribution Plots
plt.figure(1, figsize=(15,14))
n = 0 
for x in ['N','P','K','temperature','humidity','ph','rainfall']:
    n += 1
    plt.subplot(3,3,n)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    sns.histplot(df_crop[x], kde=True, bins=20)
plt.show()

# Feature and label separation
X = df_crop.drop(['label'], axis=1)
y = df_crop['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Accuracy tracking
acc = []
model = []

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier(criterion='gini', n_estimators=100, random_state=42, max_depth=None)
RandomForest.fit(X_train, y_train)
y_pred = RandomForest.predict(X_test)

ran_accuracy = accuracy_score(y_test, y_pred)
print('Random Forest Accuracy 1:', ran_accuracy * 100)
acc.append(ran_accuracy)
model.append('Random Forest')
print('Classification report:\n', classification_report(y_test, y_pred))

# Confusion Matrix for Random Forest
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,7))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted classes')
plt.ylabel('True classes')
plt.show()

# Random Forest (Experiment 1)
RandomForest = RandomForestClassifier(criterion='entropy', n_estimators=50, random_state=42, max_depth=5)
RandomForest.fit(X_train, y_train)
y_pred = RandomForest.predict(X_test)
print('Random Forest Accuracy 2:', accuracy_score(y_test, y_pred) * 100)

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
DecisonTree = DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=None)
DecisonTree.fit(X_train, y_train)
y_pred = DecisonTree.predict(X_test)
decision_accuracy = accuracy_score(y_test, y_pred)
print('DecisionTree Accuracy 1:', decision_accuracy * 100)
acc.append(decision_accuracy)
model.append('Decision Tree')
print('Classification report:\n', classification_report(y_test, y_pred))

# Confusion Matrix for Decision Tree
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,7))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted classes')
plt.ylabel('True classes')
plt.show()

# Decision Tree (Experiment 1)
DecisonTree = DecisionTreeClassifier(criterion='entropy',splitter='random',max_depth=5)
DecisonTree.fit(X_train, y_train)
y_pred = DecisonTree.predict(X_test)
print('DecisionTree Accuracy 2:', accuracy_score(y_test, y_pred) * 100)

# K Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=22, weights='distance', algorithm='auto')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn_testing_acc = metrics.accuracy_score(y_test, y_pred)
print("KNN Accuracy 1: ", knn_testing_acc * 100)
acc.append(knn_testing_acc)
model.append('KNN')
print('Classification report:\n', classification_report(y_test, y_pred))

# Confusion Matrix for KNN
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,7))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted classes')
plt.ylabel('True classes')
plt.show()

# KNN (Experiment 1)
knn = KNeighborsClassifier(n_neighbors=22, weights='uniform', algorithm='kd_tree')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("KNN Accuracy 2: ", accuracy_score(y_test, y_pred) * 100)

# Accuracy Comparison
plt.figure(figsize=[10,5], dpi=100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x=acc, y=model, palette='Blues')
plt.show()

# Sample Input
test_rc = np.array([[90, 42, 43, 20.879744, 82.002744, 6.502985, 202.935536]])
print("Test Input:", test_rc)

# Top-2 Crop Predictions (numeric)
proba = RandomForest.predict_proba(test_rc)
top_2_indices = np.argsort(proba[0])[-2:][::-1]
top_2_labels = [RandomForest.classes_[i] for i in top_2_indices]
top_2_crops = [crop_dict[label] for label in top_2_labels]

print("Top 2 Recommended Crops:")
for i in range(2):
    print(f"{top_2_crops[i]} (Class {top_2_labels[i]}): {proba[0][top_2_indices[i]]*100:.2f}% confidence")

# Save Model
with open('RDF_model.pkl', 'wb') as file:
    pickle.dump(RandomForest, file)
print("Random Forest model saved to RDF_model.pkl")

# Load Model and Predict Again
with open('RDF_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

sample_input = np.array([[90, 42, 43, 20.879744, 82.002744, 6.502985, 202.935536]])
print('\nPredicting using loaded model...')
proba_loaded = loaded_model.predict_proba(sample_input)
top_2_indices_loaded = np.argsort(proba_loaded[0])[-2:][::-1]
top_2_labels_loaded = [loaded_model.classes_[i] for i in top_2_indices_loaded]
top_2_crops_loaded = [crop_dict[label] for label in top_2_labels_loaded]

print("Top 2 Predicted Crops from Loaded Model:")
for i in range(2):
    print(f"{top_2_crops_loaded[i]} (Class {top_2_labels_loaded[i]}): {proba_loaded[0][top_2_indices_loaded[i]]*100:.2f}% confidence")

# Summary of Accuracies
print('\nRandom Forest Accuracy: ', ran_accuracy * 100)
print('DecisionTree Accuracy:', decision_accuracy * 100)
print("KNN Accuracy: ", knn_testing_acc * 100)



# sample = df_crop.groupby('label')[['region_Central India', 'region_Eastern India', 'region_North Eastern India', 'region_Northern India', 'region_Western India','region_Other']].sum()
# sample

# CI = sample.groupby('label')[['region_Central India']].sum()
# CI = CI.replace(0, pd.np.nan)
# CI=CI.dropna()
# CI

# EI = sample.groupby('label')[['region_Eastern India']].sum()
# EI = EI.replace(0, pd.np.nan)
# EI = EI.dropna()
# EI

# NEI = sample.groupby('label')[['region_North Eastern India']].sum()
# NEI = NEI.replace(0, pd.np.nan)
# NEI = NEI.dropna()
# NEI

# NI = sample.groupby('label')[['region_Northern India']].sum()
# NI = NI.replace(0, pd.np.nan)
# NI = NI.dropna()
# NI

# WI = sample.groupby('label')[['region_Western India']].sum()
# WI = WI.replace(0, pd.np.nan)
# WI=WI.dropna()
# WI

# combined_df = pd.concat([CI, EI, NEI, NI, WI])
# combined_df
