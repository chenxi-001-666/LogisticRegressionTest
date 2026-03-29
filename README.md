# Advanced Machine Learning: Logistic Regression

This repository contains the implementation for **Assignment 2** of the **WOA7015 Advanced Machine Learning** course. The project focuses on implementing a Logistic Regression model from scratch to perform binary classification on a toy dataset.

## 📂 Project Structure

* `logistic_regression.py`: Core implementation of the Logistic Regression algorithm.
* `Assignment2_25053692_ChenXi.ipynb`: Jupyter Notebook containing experimental results, data visualization, and performance analysis.
* `Train_toydata.txt`: Dataset used for training the model.
* `Test_toydata.txt`: Dataset used for evaluating the model's generalization capability.

## 🛠️ Features

1.  **Scratch Implementation**: Manual implementation of the Sigmoid function, Binary Cross-Entropy Loss, and Gradient Descent optimization.
2.  **Model Training**: Iterative weight updates using the training set.
3.  **Performance Evaluation**: Calculation of metrics such as Accuracy and Loss on the test set.
4.  **Data Visualization**: Plotting the Decision Boundary and the Learning Curve (Loss vs. Iterations).

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment. Install the dependencies using:
```bash
pip install -r requirements.txt
```

### 2. Running the Project
To run the training script:
```bash
python logistic_regression.py
```

To explore the analysis and visualizations, open the notebook:
```bash
jupyter notebook Assignment2_25053692_ChenXi.ipynb
```


### 3. Final Steps to Upload

Now, run these commands in your PyCharm terminal to push the new files to GitHub:

1.  **Stage the new files:**
    ```bash
    git add README.md requirements.txt
    ```
2.  **Commit the changes:**
    ```bash
    git commit -m "docs: add English README and requirements.txt"
    ```
3.  **Push to GitHub:**
    ```bash
    git push origin main
    ```

 ## 📝 Conclusion
This assignment demonstrates the fundamental mechanics of Logistic Regression and the importance of hyperparameter tuning (learning rate, iterations) in achieving a robust decision boundary.
