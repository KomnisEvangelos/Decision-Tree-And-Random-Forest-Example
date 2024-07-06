import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np

np.random.seed(42)
data = {
    'age': np.random.randint(18, 70, size=100),
    'income': np.random.randint(20000, 120000, size=100),
    'student': np.random.choice([0, 1], size=100),
    'credit_rating': np.random.choice([1, 2, 3], size=100),
    'education_level': np.random.choice([1, 2, 3, 4], size=100),
    'buys_computer': np.random.choice([0, 1], size=100)
}

df = pd.DataFrame(data)

X = df[['age', 'income', 'student', 'credit_rating', 'education_level']]
y = df['buys_computer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree_model = DecisionTreeClassifier()

decision_tree_model.fit(X_train, y_train)

y_pred_dt = decision_tree_model.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'Decision Tree Accuracy: {accuracy_dt * 100:.2f}%')

plt.figure(figsize=(12, 8))
tree.plot_tree(decision_tree_model, feature_names=X.columns, class_names=['Not Buy', 'Buy'], filled=True)
plt.show()

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

cv_scores = cross_val_score(random_forest_model, X, y, cv=5)
print(f'Random Forest Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%')

random_forest_model.fit(X_train, y_train)

y_pred_rf = random_forest_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf * 100:.2f}%')
