import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('CTG_cleaned.csv')
features = ['MSTV', 'MLTV', 'AC', 'AC', 'DE', 'DL', 'DP']

# Prepare the data
X = df[features]
y = df['NSP']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model
model = LogisticRegression(
    multi_class='ovr', 
    max_iter=10000,       
    tol=1e-6,              
    solver='lbfgs',               
)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient_NSP1': model.coef_[0] if len(model.classes_) > 2 else model.coef_[0],
    'Coefficient_NSP2': model.coef_[1] if len(model.classes_) > 2 else None,
    'Coefficient_NSP3': model.coef_[2] if len(model.classes_) == 3 else None
})

print("\nFeature Coefficients:")
print(feature_importance)

# Display model parameters
print(f"\nModel Classes: {model.classes_}")
print(f"Number of iterations: {model.n_iter_}")

# Test result probability of correctness
correct_predictions = (y_pred == y_test)
total_samples = len(y_test)
correct_samples = correct_predictions.sum()

# Overall probability statistics
print(f"\nTest Result Probability Analysis:")
print(f"Total test samples: {total_samples}")
print(f"Correct predictions: {correct_samples}")
print(f"Incorrect predictions: {total_samples - correct_samples}")
print(f"Probability of correctness: {correct_samples/total_samples:.4f} ({correct_samples/total_samples*100:.2f}%)")

# Confidence analysis - average probability of predicted class
predicted_class_probabilities = []
for i in range(len(y_pred)):
    predicted_class = y_pred[i]
    class_index = np.where(model.classes_ == predicted_class)[0][0]
    predicted_class_probabilities.append(y_pred_proba[i][class_index])

avg_confidence = np.mean(predicted_class_probabilities)
print(f"Average confidence in predictions: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")

# Probability distribution by class
print(f"\nProbability of correctness by NSP class:")
for nsp_class in model.classes_:
    class_mask = (y_test == nsp_class)
    if class_mask.sum() > 0:
        class_correct = correct_predictions[class_mask].sum()
        class_total = class_mask.sum()
        class_accuracy = class_correct / class_total
        print(f"NSP {nsp_class}: {class_correct}/{class_total} correct = {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")

# High confidence correct vs incorrect predictions
high_conf_threshold = 0.8
high_conf_mask = np.array(predicted_class_probabilities) > high_conf_threshold
if high_conf_mask.sum() > 0:
    high_conf_correct = correct_predictions[high_conf_mask].sum()
    high_conf_total = high_conf_mask.sum()
    print(f"\nHigh confidence predictions (>{high_conf_threshold:.1f} probability):")
    print(f"Total high confidence predictions: {high_conf_total}")
    print(f"Correct high confidence predictions: {high_conf_correct}")
    print(f"High confidence accuracy: {high_conf_correct/high_conf_total:.4f} ({high_conf_correct/high_conf_total*100:.2f}%)")