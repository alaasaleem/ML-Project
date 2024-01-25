import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Load the CSV file into a DataFrame
dataFrame = pd.read_csv(r"StressLevelDataset.csv")

# Extract features (X) and target variable (y)
X = dataFrame[['anxiety_level', 'self_esteem', 'mental_health_history', 'depression', 'headache',
               'blood_pressure', 'sleep_quality', 'breathing_problem', 'noise_level', 
               'living_conditions', 'safety', 'basic_needs', 'academic_performance', 
               'study_load', 'teacher_student_relationship', 'future_career_concerns', 
               'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']].values

y = dataFrame['stress_level'].values

# Map class values to stress level labels for visualization
class_labels = {0: 'Class 0', 1: 'Class 1', 2: 'Class '}

def evaluate_knn_model(k, X, y):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    y_pred_val = cross_val_predict(knn_model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    
    accuracy_val = accuracy_score(y, y_pred_val)
    conf_matrix = confusion_matrix(y, y_pred_val)
    tn, fp, fn, tp = conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[1, 1]
    misclassified = fp + fn + conf_matrix[0, 2] + conf_matrix[1, 2] + conf_matrix[2, 0] + conf_matrix[2, 1]
    
    return accuracy_val, conf_matrix, tn, fp, fn, tp, misclassified

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

    ax.set_title(f'Confusion Matrix (k={k}) using cross-validation', color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', color='white', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Label', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', colors='white')

# Create KNN models with different k values
k_values = [1, 3]
results = [evaluate_knn_model(k, X, y) for k in k_values]

# Plot confusion matrices and additional information
fig, axes = plt.subplots(nrows=1, ncols=len(k_values), figsize=(16, 6), facecolor='black')

for ax, (k, (accuracy_val, conf_matrix, tn, fp, fn, tp, misclassified)) in zip(axes, zip(k_values, results)):
    plot_confusion_matrix(ax, conf_matrix, f'Confusion Matrix (k={k}) using cross-validation', k, tn, fp, fn, tp, misclassified)

fig.set_facecolor('black')
plt.show()

# Plot accuracy and cross-validation error for different k values
k_values = np.arange(1, 20)
accuracies = []
cv_errors = []

for k in k_values:
    accuracy_val, _, _, _, _, _, _ = evaluate_knn_model(k, X, y)
    accuracies.append(accuracy_val)
    cv_errors.append(1 - accuracy_val)

# Create subplots for Accuracy and Cross-Validation Error
plt.figure(figsize=(12, 6), facecolor='black')

# Subplot for Accuracy
plt.subplot(1, 2, 1)
plt.plot(k_values, accuracies, label='Accuracy', linestyle='-', color='blue', linewidth=2)
plt.scatter([1, 3], [accuracies[0], accuracies[2]], color='blue', marker='o', s=100)
for k, acc in zip([1, 3], [accuracies[0], accuracies[2]]):
    plt.text(k, acc, f'{acc:.2%}', color='white', fontsize=8, ha='left', va='bottom')
plt.title('Accuracy for Different k Values', fontsize=14, color='white', fontweight='bold')
plt.xlabel('k value', fontsize=12, color='white')
plt.ylabel('Accuracy', fontsize=12, color='white')
plt.xticks(color='white', fontsize=10)
plt.yticks(color='white', fontsize=10)
plt.grid(True, color='white')
plt.gca().set_facecolor('black')
plt.legend()

# Subplot for Cross-Validation Error
plt.subplot(1, 2, 2)
plt.plot(k_values, cv_errors, label='Cross-Validation Error', linestyle='-', color='red', linewidth=2)
plt.scatter([1, 3], [cv_errors[0], cv_errors[2]], color='red', marker='o', s=100)
for k, cv_err in zip([1, 3], [cv_errors[0], cv_errors[2]]):
    plt.text(k, cv_err, f'{cv_err:.2%}', color='white', fontsize=8, ha='left', va='top')
plt.title('Cross-Validation Error for Different k Values', fontsize=14, color='white', fontweight='bold')
plt.xlabel('k value', fontsize=12, color='white')
plt.ylabel('Error', fontsize=12, color='white')
plt.xticks(color='white', fontsize=10)
plt.yticks(color='white', fontsize=10)
plt.grid(True, color='white')
plt.gca().set_facecolor('black')
plt.legend()

plt.tight_layout()
plt.show()
