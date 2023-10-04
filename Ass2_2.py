import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic dataset with two features that have different means and standard deviations
np.random.seed(123)
n_samples = 1000
X1 = np.random.normal(1, 4, n_samples)
X2 = np.random.normal(1800, 2, n_samples)
Outliers = [[10,1750],[-5,1830],[6,1840]]

X = np.vstack([X1, X2]).T
X_out = np.vstack([X,Outliers])
plt.scatter(X[:,0],X[:,1])
plt.title('Original Data')
plt.show()

plt.scatter(X_out[:,0],X_out[:,1])
plt.title('Data with outliers')
plt.show()

def min_max_scaling(df):
    out = np.empty(df.shape)
    for col in range(df.shape[1]):
        out[:,col] = (df[:,col] - df[:,col].min()) / (df[:,col].max() - df[:,col].min())
    return out

X_norm = min_max_scaling(X_out)

X_test2 = min_max_scaling(X)

plt.scatter(X_test2[:,0],X_test2[:,1])
plt.title('Original Data Minmaxed')
plt.show()

plt.scatter(X_norm[:,0],X_norm[:,1])
plt.title('Data with Outliers minmaxed')
plt.xlim(0, 1)
plt.ylim(0, 1)

# Don't mess with the limits!
plt.autoscale(False)
plt.show()

def z_score(df):
    out = np.empty(df.shape)
    for column in range(df.shape[1]):
        out[:,column] = (df[:,column] - df[:,column].mean()) / df[:,column].std() 
    return out

X_stand = z_score(X_out)

X_test = z_score(X)

plt.scatter(X_test[:,0],X_test[:,1])
plt.title('Original Data Z-scored')
plt.xlim(-4, 4)
plt.ylim(-4, 3)

# Don't mess with the limits!
plt.autoscale(False)
plt.show()

plt.scatter(X_stand[:,0],X_stand[:,1])
plt.title('Data with outliers z-scored')
plt.xlim(-4, 4)
plt.ylim(-4, 3)

# Don't mess with the limits!
plt.autoscale(False)
plt.show()

# Thus we can See that z-score normalised data is less affected by outliers




