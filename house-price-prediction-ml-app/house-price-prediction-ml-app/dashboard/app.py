import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
import os

# Load dataset
data_path = "data/kc_house_data.csv"

df = pd.read_csv(data_path)

# Create total_rooms feature
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# Use only total_rooms and price for this model
df = df[['total_rooms', 'price']].dropna()

# Define features and target
X = df['total_rooms'].values
y = df['price'].values
upper_limitX = X.mean() + (3 * X.std())
lower_limitX = X.mean() - (3 * X.std())

X = np.where(X > upper_limitX,upper_limitX,np.where(X < lower_limitX,lower_limitX,X))

X = X.reshape(-1,1)

upper_limitY = y.mean() + (3 * y.std())
lower_limitY = y.mean() - (3 * y.std())

y = np.where(y > upper_limitY,upper_limitY,np.where(y < lower_limitY,lower_limitY,y)).reshape(-1,1)

scalar = StandardScaler()
y_scaled = scalar.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_scaled, test_size=0.2, random_state=42
)





# Polynomial feature transformation
poly = PolynomialFeatures(degree=20)  # You can change degree here
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train the regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 🔽 Save model and transformers
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scalar, f)
with open("poly.pkl", "wb") as f:
    pickle.dump(poly, f)

print("✅ Model saved successfully.")
# Prepare smooth curve for visualization
X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

# Plotting

print(model.predict(X_test_poly))
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.3, label='Training Data')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Polynomial Regression (degree=20)')
plt.xlabel('Total Rooms (Bedrooms + Bathrooms)')
plt.ylabel('House Price')
plt.title('Polynomial Regression: Total Rooms vs Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

