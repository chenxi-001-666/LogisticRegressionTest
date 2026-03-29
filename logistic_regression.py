import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# --- Data Loading Function ---

def load_data(file_path):
    """
    Loads data from a .txt file. Assumes the first two columns are features (X)
    and the third column is the class label (Y).
    """
    # Load data as a NumPy array of float type
    data = np.loadtxt(file_path)

    # Extract features (X: n x p matrix, where p=2)
    X = data[:, :2]
    # Extract labels (Y: n x 1 vector)
    Y = data[:, 2].reshape(-1, 1)

    return X, Y


# --- Core Mathematical Functions (For Scratch Implementation) ---

def sigmoid(z):
    """Sigmoid activation function: g(z) = 1 / (1 + e^(-z))"""
    # Use np.clip for numerical stability to prevent overflow in np.exp
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def add_bias_col(X):
    """Adds a column of ones (the bias term X0=1) to the feature matrix X."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n = X.shape[0]
    # Create the column of ones
    bias_col = np.ones((n, 1))
    # Concatenate the bias column with the original features
    X_with_bias = np.hstack((bias_col, X))
    return X_with_bias


# --- Required Functions for Scratch Implementation (Task 1) ---

def CalcObj(XTrain, YTrain, wHat):
    """
    Calculates the Cross-Entropy Loss (Objective Function J(w)).

    XTrain is an n x p matrix. wHat is a (p+1) x 1 vector.
    The return value 'obj' is the value of the objective function.
    """
    X_bias = add_bias_col(XTrain)
    z = X_bias.dot(wHat)
    h = sigmoid(z)

    # Clip probabilities to prevent log(0)
    h = np.clip(h, 1e-10, 1 - 1e-10)

    # Calculate mean Cross-Entropy Loss
    loss = -np.mean(YTrain * np.log(h) + (1 - YTrain) * np.log(1 - h))

    return loss


def CalcGrad(XTrain, YTrain, wHat):
    """
    Calculates the gradient of the objective function J(w) w.r.t wHat.

    The return value 'grad' is a (p+1) x 1 vector.
    """
    X_bias = add_bias_col(XTrain)
    z = X_bias.dot(wHat)
    h = sigmoid(z)

    error = h - YTrain

    # Calculate mean gradient
    grad = (1.0 / X_bias.shape[0]) * X_bias.T.dot(error)

    return grad


def UpdateParams(weight, grad, lr):
    """
    Updates parameters using Gradient Descent: w_new = w_old - lr * grad.

    'lr' is the step size, set to 0.01.
    The return value 'wHat' is the updated parameter estimate.
    """
    new_weight = weight - lr * grad
    return new_weight


def CheckConvg(oldObj, newObj, tol):
    """
    Checks if the objective function value has converged.

    'tol' is the convergence tolerance, set to 0.001.
    Convergence condition: |oldObj - newObj| / |oldObj| < tol.
    """
    if oldObj is None:
        return False

    # Calculate the relative change
    relative_change = np.abs(oldObj - newObj) / np.abs(oldObj)

    # hasConverged is a boolean value
    return relative_change < tol


def GradientDescent(XTrain, YTrain, lr=0.01, tol=0.001, max_iter=10000):
    """
    Trains the logistic regression model using Gradient Descent.

    'lr' is set to 0.01. 'tol' is set to 0.001.
    'objVals' is a vector containing the objective value at each iteration.
    """
    # Initialize parameters wHat. p=2, so wHat is 3x1 (w0, w1, w2)
    p = XTrain.shape[1]
    # 1. weights is randomly assigned
    wHat = np.random.randn(p + 1, 1)

    objVals = []
    oldObj = None

    print(f"Starting Training (lr={lr}, tol={tol})")
    for i in range(max_iter):
        # 2. calculate the output & 3. calculate the loss
        newObj = CalcObj(XTrain, YTrain, wHat)
        objVals.append(newObj)

        # Check convergence
        if CheckConvg(oldObj, newObj, tol):
            print(f"✅ Converged at iteration {i + 1}. Final Loss: {newObj:.4f}")
            break

        # 4. calculate the gradient
        grad = CalcGrad(XTrain, YTrain, wHat)

        # 5. based on the gradient, update the weight
        wHat = UpdateParams(wHat, grad, lr)

        oldObj = newObj

        if (i + 1) % 1000 == 0:
            print(f"Iteration {i + 1}/{max_iter}, Current Loss: {newObj:.4f}")
    else:
        # 6. repeat stepl to step5 again until loss is minimized
        print(f"⚠️ Reached max iterations {max_iter}, did not converge. Final Loss: {newObj:.4f}")

    return wHat, np.array(objVals)


def PredictLabels(XTest, YTest, wHat):
    """
    Uses the trained wHat to make predictions and calculate the number of errors.

    'yHat' is the predicted class labels.
    'numErrors' is the number of misclassified examples.
    """
    X_bias = add_bias_col(XTest)

    # Calculate probability h(x)
    z = X_bias.dot(wHat)
    h = sigmoid(z)

    # Convert probability to class label (0 or 1)
    yHat = (h >= 0.5).astype(int)

    # Calculate the number of misclassified examples
    numErrors = np.sum(yHat != YTest)

    return yHat, numErrors


# --- PyTorch Implementation (Task 2) ---

class LogisticRegressionPyTorch(nn.Module):
    """PyTorch module for Logistic Regression."""

    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        # The output of linear layer is passed through Sigmoid
        return torch.sigmoid(self.linear(x))


def train_and_test_pytorch(XTrain_np, YTrain_np, XTest_np, YTest_np, lr=0.01):
    """Trains and tests the model using built-in PyTorch functions."""
    # 1. Data Conversion: NumPy array -> PyTorch Tensor (float32)
    XTrain_tensor = torch.from_numpy(XTrain_np).float()
    YTrain_tensor = torch.from_numpy(YTrain_np).float()
    XTest_tensor = torch.from_numpy(XTest_np).float()
    YTest_tensor = torch.from_numpy(YTest_np).float()

    # 2. Model, Loss Function, and Optimizer Initialization
    input_size = XTrain_tensor.shape[1]
    model = LogisticRegressionPyTorch(input_size)

    # Binary Cross Entropy Loss
    criterion = nn.BCELoss(reduction='mean')
    # SGD Optimizer (Gradient Descent). lr is set to 0.01.
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 3. Training Loop
    num_epochs = 10000
    print(f"Starting PyTorch Training (lr={lr}, epochs={num_epochs})")

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(XTrain_tensor)
        loss = criterion(outputs, YTrain_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters

        if (epoch + 1) % 1000 == 0:
            print(f'PyTorch Training - Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 4. Testing the Model
    with torch.no_grad():  # Disable gradient calculation during testing
        outputs = model(XTest_tensor)

        # Convert probability to predicted label (0 or 1)
        yHat_tensor = (outputs >= 0.5).int()

        # Calculate the number of errors
        numErrors = torch.sum(yHat_tensor != YTest_tensor.int()).item()

    return yHat_tensor.numpy(), numErrors


# --- Main Execution Logic ---

if __name__ == "__main__":
    # Load data from Train_toydata.txt and Test_toydata.txt
    XTrain, YTrain = load_data("Train_toydata.txt")
    XTest, YTest = load_data("Test_toydata.txt")

    print("=" * 50)
    print("--- Task 1: Logistic Regression from Scratch (Numpy) ---")
    print("=" * 50)

    # Train the model
    wHat_scratch, objVals_scratch = GradientDescent(XTrain, YTrain, lr=0.01, tol=0.001)

    # Test the model
    yHat_scratch, numErrors_scratch = PredictLabels(XTest, YTest, wHat_scratch)

    m_test = XTest.shape[0]
    accuracy_scratch = (m_test - numErrors_scratch) / m_test * 100

    print("\n[Summary - Scratch Implementation]")
    print(f"Final parameters wHat (including bias): \n{wHat_scratch.flatten()}")
    print(f"Test samples (m): {m_test}")
    print(f"Number of errors (numErrors): {numErrors_scratch}")
    print(f"Test Accuracy: {accuracy_scratch:.2f}%")

    print("\n" + "=" * 50)
    print("--- Task 2: Logistic Regression using Built-in PyTorch ---")
    print("=" * 50)

    # Train and test the PyTorch model
    yHat_pytorch, numErrors_pytorch = train_and_test_pytorch(XTrain, YTrain, XTest, YTest, lr=0.01)

    accuracy_pytorch = (m_test - numErrors_pytorch) / m_test * 100

    print("\n[Summary - PyTorch Built-in]")
    print(f"Test samples (m): {m_test}")
    print(f"Number of errors (numErrors): {numErrors_pytorch}")
    print(f"Test Accuracy: {accuracy_pytorch:.2f}%")

    print("\n" + "=" * 50)
    print("--- Task 3: Comparison and Report Data ---")
    print("Please copy the following table to your report, then compare the accuracy between your own functions and the built-in PyTorch function, and explain the reason for any discrepancy:")

    # Report table data
    report_table = f"""
| Method | Num Errors | Accuracy |
| :--- | :---: | :---: |
| Scratch (Numpy) | {numErrors_scratch} | {accuracy_scratch:.2f}% |
| Built-in PyTorch | {numErrors_pytorch} | {accuracy_pytorch:.2f}% |
"""
    print(report_table)
