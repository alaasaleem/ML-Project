import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file into a DataFrame
dataFrame = pd.read_csv(r"StressLevelDataset.csv")

# Extract features (X) and target variable (y)
X = dataFrame[['self_esteem', 'sleep_quality', 'noise_level', 'living_conditions',
               'academic_performance', 'study_load', 'teacher_student_relationship',
               'future_career_concerns', 'social_support', 'peer_pressure',
               'extracurricular_activities', 'bullying', 'stress_level']].values

y = dataFrame['depression'].values

# Split data into training set, validation set, and testing set
X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=220, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=220, shuffle=False)

# Define the KNN model with Euclidean distance and k=1
knn_model_k1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn_model_k1.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred_k1 = knn_model_k1.predict(X_val)

# Evaluate the performance for k=1
accuracy_k1 = accuracy_score(y_val, y_val_pred_k1)
print(f'Accuracy for k=1: {accuracy_k1}')
print('Classification Report for k=1:')
print(classification_report(y_val, y_val_pred_k1))

# Define the KNN model with Euclidean distance and k=3
knn_model_k3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_model_k3.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred_k3 = knn_model_k3.predict(X_val)

# Evaluate the performance for k=3
accuracy_k3 = accuracy_score(y_val, y_val_pred_k3)
print(f'\nAccuracy for k=3: {accuracy_k3}')
print('Classification Report for k=3:')
print(classification_report(y_val, y_val_pred_k3))
