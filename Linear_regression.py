import numpy as np

def linear_regression(X, y):
    # Add a column of ones for the bias term (intercept)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Calculate the regression coefficients using the normal equation
    # The normal equation is: beta = (X.T * X)^-1 * X.T * y
    beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    return beta

def predict(X, beta):
    # Add a column of ones for the bias term (intercept)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    # Calculate the predicted values using the learned coefficients (beta)
    return X_b.dot(beta)

def manual_standard_scaler(data):

    # Compute the mean and standard deviation for each feature (column)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Apply z-score normalization: (data - mean) / std
    scaled_data = (data - mean) / std
    return scaled_data

def mean_absolute_error(y_true, y_pred):
    """Calculate the Mean Absolute Error (MAE)."""
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    """Calculate the Root Mean Squared Error (RMSE)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
