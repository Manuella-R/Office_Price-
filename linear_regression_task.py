# linear_regression_task.py

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.
    
    Parameters:
    - y_true: Actual target values
    - y_pred: Predicted values
    
    Returns:
    - MSE value
    """
    return np.mean((y_true - y_pred) ** 2)

# Define Gradient Descent function
def gradient_descent(x, y, m_init, c_init, learning_rate=0.01, epochs=10):
    """
    Perform gradient descent to fit a line to the data with scaled features.
    
    Parameters:
    - x: Feature values (office sizes)
    - y: Target values (office prices)
    - m_init, c_init: Initial values for slope (m) and intercept (c)
    - learning_rate: Learning rate for gradient descent
    - epochs: Number of training iterations
    
    Returns:
    - m: Final slope
    - c: Final intercept
    """
    m, c = m_init, c_init
    n = len(x)
    
    for epoch in range(epochs):
        y_pred = m * x + c  # Predicted values
        m_grad = (-2 / n) * np.sum(x * (y - y_pred))  # Gradient of m
        c_grad = (-2 / n) * np.sum(y - y_pred)  # Gradient of c
        
        # Update weights
        m -= learning_rate * m_grad
        c -= learning_rate * c_grad
        
        # Calculate and print the Mean Squared Error for each epoch
        error = mean_squared_error(y, y_pred)
        print(f"Epoch {epoch+1}: m={m:.4f}, c={c:.4f}, MSE={error:.4f}")
    
    return m, c

# Load dataset
data = pd.read_csv('nairobi_office_prices.csv')  # Ensure this file is in the same directory

# Use SIZE as the feature (x) and PRICE as the target (y)
x = data['SIZE'].values  # Feature
y = data['PRICE'].values  # Target

# Feature Scaling (Normalization)
x_mean, x_std = np.mean(x), np.std(x)
y_mean, y_std = np.mean(y), np.std(y)

x = (x - x_mean) / x_std
y = (y - y_mean) / y_std

# Set random initial values for m and c
m_init = np.random.rand()
c_init = np.random.rand()

# Train the model for 10 epochs
m, c = gradient_descent(x, y, m_init, c_init, learning_rate=0.01, epochs=10)

# Rescale line of best fit back to original scale
x_original = (x * x_std) + x_mean
y_pred_scaled = m * x + c
y_pred_original = (y_pred_scaled * y_std) + y_mean

# Plot the line of best fit
plt.scatter(x_original, (y * y_std) + y_mean, color='blue', label='Data points')
plt.plot(x_original, y_pred_original, color='red', label='Line of Best Fit')
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price')
plt.legend()
plt.title('Linear Regression Line of Best Fit')
plt.show()

# Predict office price for a size of 100 sq. ft. with scaling adjustments
size = 100
size_scaled = (size - x_mean) / x_std
predicted_price_scaled = m * size_scaled + c
predicted_price_original = (predicted_price_scaled * y_std) + y_mean
print(f"The predicted office price for a 100 sq. ft. office is: {predicted_price_original:.2f}")
