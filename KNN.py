import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Load the CSV file into a DataFrame
dataFrame = pd.read_csv(r"StressLevelDataset.csv")

# Extract features (X) and target variable (y)
X = dataFrame[['anxiety_level', 'self_esteem', 'mental_health_history', 'depression', 'headache',
               'blood_pressure', 'sleep_quality', 'breathing_problem', 'noise_level', 
               'living_conditions', 'safety', 'basic_needs', 'academic_performance', 
               'study_load', 'teacher_student_relationship', 'future_career_concerns', 
               'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']].values

y = dataFrame['stress_level'].values

# Train-validation-test split (60:20:20))
X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=220, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=220, shuffle=False)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
# Map class values to stress level labels for visualization
class_labels = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2'}

def evaluate_knn_model(k, X_train, y_train, X_val, y_val):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred_val = knn_model.predict(X_val)
    
    accuracy_val = accuracy_score(y_val, y_pred_val)
    precision_val = precision_score(y_val, y_pred_val, average='weighted')
    recall_val = recall_score(y_val, y_pred_val, average='weighted')
    f1_val = f1_score(y_val, y_pred_val, average='weighted')
    
    accuracy_train = accuracy_score(y_train, knn_model.predict(X_train))
    
    return accuracy_val, precision_val, recall_val, f1_val, accuracy_train, knn_model

def plot_confusion_matrix(ax, conf_matrix, title, k, tn, fp, fn, tp, misclassified):
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Greys",
        xticklabels=class_labels.values(),
        yticklabels=class_labels.values(),
        linewidths=.5,
        square=True,
        cbar=False,
        ax=ax,
        annot_kws={"color": 'white', "fontfamily": "serif", "fontsize": 12, "style": "italic", "weight": "bold", "bbox": dict(boxstyle="round", alpha=0.1, facecolor='black')}
    )
    
    ax.text(1.5, -0.2, f'Correctly Classified: {tp + tn + conf_matrix[2, 2]}', horizontalalignment='center', verticalalignment='bottom', color='green', fontweight='bold')
    ax.text(1.5, -0.3, f'Misclassified: {misclassified}', horizontalalignment='center', verticalalignment='bottom', color='red', fontweight='bold')
    ax.text(1.5, -0.4, f'Classification Error: {misclassified / (tp + fp + tn + fn):.2%}', horizontalalignment='center', verticalalignment='bottom', color='white', fontweight='bold')

    ax.set_title(f'Confusion Matrix (k={k})', color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', color='white', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Label', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', colors='white')

# Create KNN models with different k values
k_values = [1, 3]
results = [evaluate_knn_model(k, X_train_scaled, y_train, X_val_scaled, y_val) for k in k_values]

# Save accuracies in an array
accuracies_array = []

for k, (accuracy_val, _, _, _, _, _) in zip(k_values, results):
    accuracies_array.append(accuracy_val)

# Plot both confusion matrices in the same figure
fig, axes = plt.subplots(nrows=1, ncols=len(k_values), figsize=(16, 6), facecolor='black')

for i, (k, (_, _, _, _, _, knn_model)) in enumerate(zip(k_values, results)):
    y_pred_val = knn_model.predict(X_val_scaled)
    
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_val, y_pred_val)
    tn, fp, fn, tp = conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[1, 1]
    misclassified = fp + fn
    
    plot_confusion_matrix(axes[i], conf_matrix, f'Confusion Matrix (k={k})', k, tn, fp, fn, tp, misclassified)
    axes[i].set_facecolor('black')

plt.tight_layout()
plt.show()
training_errors = []
for k, (accuracy_val, _, _, _, accuracy_train, _) in zip(k_values, results):
    y_pred_val = _.predict(X_val_scaled)
    conf_matrix = confusion_matrix(y_val, y_pred_val)
    tn, fp, fn, tp = conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[1, 1]
    misclassified = fp + fn

    print(f'Evaluation Metrics for k={k}:')
    print(f'Accuracy (Validation): {accuracy_val:.2%}')
    print(f'Accuracy (Training): {accuracy_train:.2%}')  # Include training accuracy here
    print(f'Classification Error: {misclassified / (tp + fp + tn + fn):.2%}')
    print(f'Misclassified: {misclassified}')
    print(f'Correctly Classified: {tp + tn + conf_matrix[2, 2]}')
    print(f'Training Error: {1 - accuracy_train:.2%}')  # Use accuracy_train here
    print(f'Validation Error: {1 - accuracy_val:.2%}')
    print('\n')


# Plot accuracy, training error, and validation error for different k values
k_values = np.arange(1, 20)
accuracies = []

validation_errors = []

for k in k_values:
    accuracy_val, accuracy_train, _, _, _, _ = evaluate_knn_model(k, X_train_scaled, y_train, X_val_scaled, y_val)
    accuracies.append(accuracy_val)
    training_errors.append(1 - accuracy_train)  # Training error is 1 - Training Accuracy
    validation_errors.append(1 - accuracy_val)  # Validation error is 1 - Validation Accuracy

# Plot Accuracy, Training Error, and Validation Error
plt.figure(figsize=(16, 8), facecolor='black')
specific_k_values = [1, 3] 
# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(k_values, accuracies, label='Accuracy', linestyle='-', color='blue', linewidth=2)
plt.scatter(specific_k_values, [accuracies[0], accuracies[2]], color='blue', marker='o', s=100)
for k, acc in zip(specific_k_values, [accuracies[0], accuracies[2]]):
    plt.text(k, acc, f'K={k}: {acc:.2%}', color='white', fontsize=8, ha='left', va='bottom', bbox=dict(facecolor='black', edgecolor='white', boxstyle='round', alpha=0.5))
plt.title('Accuracy for Different k Values', fontsize=14, color='white', fontweight='bold')
plt.xlabel('k value', fontsize=12, color='white')
plt.ylabel('Accuracy', fontsize=12, color='white')
plt.xticks(color='white', fontsize=10)
plt.yticks(color='white', fontsize=10)
plt.grid(True, color='white')
plt.gca().set_facecolor('black')
plt.legend()

# Create a subplot for Training Error and Validation Error
plt.subplot(1, 2, 2)
plt.plot(k_values, training_errors, label='Training Error', linestyle='-', color='orange', linewidth=2)
plt.plot(k_values, validation_errors, label='Validation Error', linestyle='-', color='green', linewidth=2)
plt.scatter(specific_k_values, [training_errors[0], training_errors[2]], color='orange', marker='o', s=100)
plt.scatter(specific_k_values, [validation_errors[0], validation_errors[2]], color='green', marker='o', s=100)


for k, error in zip(specific_k_values, [training_errors[0], training_errors[2]]):
    plt.text(k, error, f'K={k}: {error:.2%}', color='white', fontsize=8, ha='left', va='bottom', bbox=dict(facecolor='black', edgecolor='white', boxstyle='round', alpha=0.5))
    
for k, error in zip(specific_k_values, [validation_errors[0], validation_errors[2]]):
    plt.text(k, error, f'K={k}: {error:.2%}', color='white', fontsize=8, ha='left', va='bottom', bbox=dict(facecolor='black', edgecolor='white', boxstyle='round', alpha=0.5))

plt.title('Training and Validation Errors for Different k Values', fontsize=14, color='white', fontweight='bold')
plt.xlabel('k value', fontsize=12, color='white')
plt.ylabel('Error', fontsize=12, color='white')
plt.xticks(color='white', fontsize=10)
plt.yticks(color='white', fontsize=10)
plt.grid(True, color='white')
plt.gca().set_facecolor('black')
plt.legend()

plt.tight_layout()
plt.show()
