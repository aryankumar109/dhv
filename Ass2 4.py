import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Create a synthetic dataset with two features that have different means and standard deviations
np.random.seed(123)
n_samples = 1200
X1 = np.random.normal(1, 4, n_samples)
X2 = np.random.normal(1800, 2, n_samples)
X11 = np.random.normal(10, 2, n_samples)
X22 = np.random.normal(1800, 3, n_samples)
x1 = np.vstack([X1, X2]).T
x2 = np.vstack([X11, X22]).T
X = np.vstack([x1,x2])
y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])

plt.scatter(X[:n_samples,0],X[:n_samples,1])
plt.scatter(X[n_samples:,0],X[n_samples:,1])
plt.title('Before Standardization')
plt.show()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Train a logistic regression classifier on the normalized data
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy before standardization: {acc:.4f}")


def z_score(df):
    for column in df.columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std() 
    return df

xdf = pd.DataFrame(X)
z_score(xdf)

plt.scatter(xdf[0][:n_samples],xdf[1][:n_samples])
plt.scatter(xdf[0][n_samples:],xdf[1][n_samples:])
plt.title('After standardization')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(xdf, y, test_size=0.2)
# Train a logistic regression classifier on the normalized data
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy after standardization: {acc:.4f}")
