# ============================================
# TASK 7 - SUPPORT VECTOR MACHINES (SVM)
# Updated Version - Memory Safe
# ============================================

# 1️⃣ Import Libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA


print("\n========== TASK 7 : SUPPORT VECTOR MACHINE ==========\n")

# =====================================================
# 2️⃣ Load Dataset
# =====================================================
print("Loading dataset...\n")
data = load_breast_cancer()
X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Classes:", np.unique(y), "\n")


# =====================================================
# 3️⃣ Train-Test Split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 4️⃣ Feature Scaling
# =====================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =====================================================
# 5️⃣ Linear SVM
# =====================================================
print("Training Linear SVM...\n")
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)

pred_linear = linear_svm.predict(X_test)
print("Linear Accuracy:", accuracy_score(y_test, pred_linear), "\n")


# =====================================================
# 6️⃣ RBF SVM
# =====================================================
print("Training RBF SVM...\n")
rbf_svm = SVC(kernel='rbf')
rbf_svm.fit(X_train, y_train)

pred_rbf = rbf_svm.predict(X_test)
print("RBF Accuracy:", accuracy_score(y_test, pred_rbf), "\n")


# =====================================================
# 7️⃣ Cross Validation
# =====================================================
print("Cross Validation...\n")
cv_scores = cross_val_score(rbf_svm, X, y, cv=5)
print("CV Scores:", cv_scores)
print("Average CV Accuracy:", cv_scores.mean(), "\n")


# =====================================================
# 8️⃣ Hyperparameter Tuning
# =====================================================
print("Hyperparameter tuning...\n")

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

best_model = grid.best_estimator_
pred_best = best_model.predict(X_test)

print("Best Model Accuracy:", accuracy_score(y_test, pred_best), "\n")


# =====================================================
# 9️⃣ Confusion Matrix
# =====================================================
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_best))
print("\nClassification Report:\n", classification_report(y_test, pred_best))


# =====================================================
# 🔟 Decision Boundary (Memory Safe Version)
# =====================================================
print("\nPlotting Decision Boundary (Safe Mode)...\n")

# Use PCA to reduce to 2D
pca = PCA(n_components=2)
X_2D = pca.fit_transform(X)

# Use only small subset to avoid memory error
X_small = X_2D[:200]
y_small = y[:200]

model_2D = SVC(kernel='rbf')
model_2D.fit(X_small, y_small)

# Small grid
x_min, x_max = X_small[:, 0].min() - 1, X_small[:, 0].max() + 1
y_min, y_max = X_small[:, 1].min() - 1, X_small[:, 1].max() + 1

h = 0.5   # bigger step size = less memory

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

Z = model_2D.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_small[:, 0], X_small[:, 1], c=y_small, edgecolors='k')
plt.title("SVM Decision Boundary (2D PCA)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

print("\n========== TASK COMPLETED SUCCESSFULLY ==========\n")