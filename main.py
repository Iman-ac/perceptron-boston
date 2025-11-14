import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import urllib.request 

"""Load Boston Housing dataset"""
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
try:
    df = pd.read_csv(url)
    print("Boston CSV loaded successfully!")
    print("Data shape:", df.shape)  
    X = df[['rm', 'age']].values  # RM=اتاق‌ها (مثل مساحت), AGE=سن
    y = df['medv'].values.reshape(-1, 1)  # MEDV=قیمت (هزار دلار)
    features = ['RM', 'AGE']
    dataset_name = "Boston"
except Exception as e:
    print("خطا در لود CSV:", str(e))
    
 
    raise e

"""Initial statistics"""
print("همبستگی ویژگی‌ها با y:")
df_corr = pd.concat([pd.DataFrame(X, columns=features), pd.DataFrame(y, columns=['Price'])], axis=1)
print(df_corr.corr()['Price'])

"""Split data into train/test (80/20)"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Normalization"""
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

print("\nآماده برای آموزش! (train shape:", X_train_scaled.shape, ")")

class PerceptronRegressor:
    def __init__(self, learning_rate=0.001, n_epochs=2000):
        self.lr = learning_rate
        self.epochs = n_epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
        return self
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

"""Training"""
model = PerceptronRegressor(learning_rate=0.001, n_epochs=2000)
model.fit(X_train_scaled, y_train_scaled.ravel())

"""Prediction and evaluation"""
y_train_pred_scaled = model.predict(X_train_scaled)
y_test_pred_scaled = model.predict(X_test_scaled)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()

print("MSE Train:", mean_squared_error(y_train, y_train_pred))
print("MSE Test:", mean_squared_error(y_test, y_test_pred))
print("وزن‌ها (w1=RM, w2=AGE):", model.weights)
print("Bias:", model.bias)

"""Plot 3D"""
fig = plt.figure(figsize=(12, 5))

"""Real Data"""
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_train[:, 0], X_train[:, 1], y_train, c='blue', alpha=0.6)
ax1.set_xlabel('RM (Rooms)')
ax1.set_ylabel('AGE')
ax1.set_zlabel('Price (k$)')
ax1.set_title('Real Data (3D) - Boston')

"""Real + Predicted Plane"""
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_train[:, 0], X_train[:, 1], y_train, c='blue', alpha=0.6)

"""Meshgrid for plane"""
xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 20),
                     np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 20))
xx_scaled = scaler_X.transform(np.c_[xx.ravel(), yy[0,0] * np.ones_like(xx.ravel())])[:, 0].reshape(xx.shape)
yy_scaled = scaler_X.transform(np.c_[xx[0,0] * np.ones_like(yy.ravel()), yy.ravel()])[:, 1].reshape(yy.shape)
zz_scaled = model.predict(np.c_[xx_scaled.ravel(), yy_scaled.ravel()]).reshape(xx.shape)
zz = scaler_y.inverse_transform(zz_scaled.reshape(-1, 1)).reshape(xx.shape)

ax2.plot_surface(xx, yy, zz, alpha=0.5, cmap='viridis')
ax2.set_xlabel('RM (Rooms)')
ax2.set_ylabel('AGE')
ax2.set_zlabel('Price (k$)')
ax2.set_title('Real + Predicted Plane (3D)')

plt.tight_layout()
plt.savefig('3d_boston_plot.png', dpi=300, bbox_inches='tight')
plt.show()