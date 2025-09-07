# Logistic Regression from Scratch: Diabetes Prediction

This project is a from-scratch implementation of the Logistic Regression algorithm using only **Python** and **NumPy**. The goal is to build a binary classification model to predict the onset of diabetes based on diagnostic measures, without relying on high-level machine learning libraries like scikit-learn.

The project is contained within the `ML_Algorithms.ipynb` Jupyter Notebook.

---

## 🎯 Project Overview

This model is built to solve a real-world classification problem using the **PIMA Indians Diabetes Dataset**. By implementing each component of the algorithm manually, the project serves as a deep dive into the inner workings of one of the most fundamental algorithms in machine learning.

### Key Implementation Steps:
1.  **Data Loading and Exploration:** The dataset is loaded using Pandas and analyzed to understand its structure and features.
2.  **Feature Normalization:** Input features are scaled to a common range to ensure the gradient descent algorithm converges efficiently.
3.  **Sigmoid Function:** The core activation function for logistic regression is implemented to map outputs to a probability score between 0 and 1.
4.  **Cost Function:** The logistic loss function is coded to measure the performance of the model by quantifying the "cost" or error of its predictions.
6.  **Gradient Descent(TO be Implimented):** The optimization algorithm is built from scratch to iteratively update the model's parameters (`w` and `b`) and minimize the cost function.
7.  **Prediction(TO be Implimented):** A function is created to make predictions on new data using the optimized parameters.

---

## 🛠️ Technologies Used

*   **Python:** The core programming language.
*   **NumPy:** For all numerical operations and vectorization.
*   **Pandas:** For data loading and manipulation.
*   **Matplotlib:** For data visualization.

---

## 🚀 How to Run

1.  Clone the repository:
    ```
    git clone https://github.com/Meetbhoi/ML_Learning_Project.git
    ```
2.  Navigate to the project directory:
    ```
    cd ML_Learning_Project
    ```
3.  Ensure you have the required libraries installed:
    ```
    pip install numpy pandas matplotlib jupyter
    ```
4.  Launch Jupyter Notebook:
    ```
    jupyter notebook
    ```
5.  Open the `ML_Algorithms.ipynb` file and run the cells.
