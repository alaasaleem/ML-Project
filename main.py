import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file into a DataFrame
dataFrame = pd.read_csv(r"smoking.csv")

# Convert 'gender' column to numeric values (assuming 'F' becomes 0 and 'M' becomes 1)
dataFrame['gender'] = dataFrame['gender'].map({'F': 0, 'M': 1})

# Convert 'tartar' and 'oral' columns to numeric values (assuming 'N' becomes 0 and 'Y' becomes 1)
dataFrame['tartar'] = dataFrame['tartar'].map({'N': 0, 'Y': 1})
dataFrame['oral'] = dataFrame['oral'].map({'N': 0, 'Y': 1})

# Extract features (X) and target variable (y)
X = dataFrame[['ID', 'gender', 'age', 'height(cm)', 'weight(kg)', 'waist(cm)', 
               'eyesight(left)', 'eyesight(right)', 'hearing(left)', 'hearing(right)', 
               'systolic', 'relaxation', 'fasting blood sugar', 'Cholesterol', 
               'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein', 
               'serum creatinine', 'AST', 'ALT', 'Gtp', 'oral', 'dental caries', 'tartar']].values

y = dataFrame['smoking'].values

# Split data into training set, validation set, and testing set
X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=11138, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=11138, shuffle=False)


import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Function to evaluate k-NN with different distance metrics
def evaluate_knn(X_train, y_train, X_val, y_val, k, metric):
    # Create k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)

    # Train the model
    knn.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = knn.predict(X_val)

    # Evaluate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Accuracy for k={k} and metric={metric}: {accuracy:.4f}')

    # Display classification report
    print(classification_report(y_val, y_pred))

    # Return the accuracy for plotting
    return accuracy

# Initialize a list to store accuracies for different values of k and metrics
accuracies_manhattan = []
accuracies_euclidean = []

# Evaluate for k=1
accuracy_k1_manhattan = evaluate_knn(X_train, y_train.ravel(), X_val, y_val.ravel(), k=1, metric='manhattan')
accuracy_k1_euclidean = evaluate_knn(X_train, y_train.ravel(), X_val, y_val.ravel(), k=1, metric='euclidean')

accuracies_manhattan.append(accuracy_k1_manhattan)
accuracies_euclidean.append(accuracy_k1_euclidean)

# Evaluate for k=3
accuracy_k3_manhattan = evaluate_knn(X_train, y_train.ravel(), X_val, y_val.ravel(), k=3, metric='manhattan')
accuracy_k3_euclidean = evaluate_knn(X_train, y_train.ravel(), X_val, y_val.ravel(), k=3, metric='euclidean')

accuracies_manhattan.append(accuracy_k3_manhattan)
accuracies_euclidean.append(accuracy_k3_euclidean)

# Plot the performance
k_values = [1, 3]
plt.bar(np.array(k_values) - 0.2, accuracies_manhattan, width=0.4, label='Manhattan', color='blue')
plt.bar(np.array(k_values) + 0.2, accuracies_euclidean, width=0.4, label='Euclidean', color='green')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('k-NN Performance with Different Distance Metrics')
plt.xticks(k_values)
plt.legend()
plt.ylim(0, 1)
plt.show()
