import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

# 1. INPUT: Load the Iris Dataset (150 samples, 4 features, 3 classes)
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y]

print("--- Dataset Overview (First 5 Rows) ---")
print(df.head())
print(f"\nTotal Samples: {X.shape[0]} | Features: {X.shape[1]} | Classes: {list(target_names)}\n")

# 2. STRUCTURAL INTEGRITY: Shuffle & Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
)

# 3. THE GATEKEEPER RULE: Feature Scaling (Mean = 0, Variance = 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. TUNING THE ENGINE: Check Error Rates for K values (Elbow Check)
print("--- Elbow Method Check (Error Rate vs K) ---")
for k in [1, 3, 5, 7, 9]:
    temp_model = KNeighborsClassifier(n_neighbors=k)
    temp_model.fit(X_train_scaled, y_train)
    temp_pred = temp_model.predict(X_test_scaled)
    error_rate = 1 - accuracy_score(y_test, temp_pred)
    print(f"K = {k} -> Error Rate: {error_rate:.4f}")

# 5. THE ALGORITHM: Train KNN Model (K = 5)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Predict on unseen test data
predictions = model.predict(X_test_scaled)

# 6. OUTPUT VALIDATION: Confusion Matrix & F1 Score
print("\n--- Model Evaluation (K=5) ---")
acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')
cm = confusion_matrix(y_test, predictions)

print(f"Accuracy Score : {acc * 100:.2f}%")
print(f"Weighted F1 Score : {f1:.4f}\n")

print("Confusion Matrix:")
print(pd.DataFrame(cm, index=target_names, columns=target_names))

print("\nDetailed Classification Report (Precision, Recall, F1):")
print(classification_report(y_test, predictions, target_names=target_names))

# 7. LIVE TEST: Predict on a new custom flower measurement
sample_flower = [[5.1, 3.5, 1.4, 0.2]]  # [Sepal Length, Sepal Width, Petal Length, Petal Width]
sample_scaled = scaler.transform(sample_flower)
predicted_class = target_names[model.predict(sample_scaled)[0]]
print(f"Sample Prediction for {sample_flower[0]} -> {predicted_class.upper()}")