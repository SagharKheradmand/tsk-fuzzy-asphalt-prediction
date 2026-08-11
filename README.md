# TSK Fuzzy System for Asphalt Property Prediction

## Overview

This project implements a Takagi-Sugeno-Kang (TSK) fuzzy inference system for predicting important properties of asphalt mixtures.

The main objective is to build an interpretable nonlinear regression model that combines fuzzy clustering with local linear models. Fuzzy C-Means (FCM) clustering is used to identify regions in the input space, and each cluster is converted into a fuzzy rule.

The system predicts four asphalt properties independently:

- Stability
- Flow
- ITSM at 20°C
- ITSM at 30°C

The complete pipeline includes data preprocessing, fuzzy clustering, TSK rule construction, parameter estimation, model evaluation, visualization, and interactive prediction.


## Main Features

- Takagi-Sugeno-Kang fuzzy inference system
- Fuzzy C-Means clustering implemented from scratch
- Gaussian antecedent membership functions
- First-order linear rule consequents
- Weighted Least Squares estimation
- Ridge regularization
- Separate models for four target properties
- Train/test evaluation
- RMSE, MAE, MAPE, and R² metrics
- Automatic selection of the number of fuzzy rules
- Prediction visualizations
- Interactive command-line prediction


## Problem Description

The goal of this project is to predict asphalt mixture properties from a set of continuous input variables.

Given an input vector:

`x = [x1, x2, ..., xn]`

the model estimates a target property using a collection of fuzzy IF-THEN rules.

A typical first-order TSK rule can be expressed as:

`IF x1 is A1 AND x2 is A2 AND ... AND xn is An`

`THEN y = b0 + b1*x1 + b2*x2 + ... + bn*xn`

Each rule represents a local linear model. The final prediction is obtained by combining the outputs of all active rules according to their normalized firing strengths.


## Prediction Targets

Four independent regression models are trained.

| Target | Description |
| --- | --- |
| Stability | Marshall stability of the asphalt mixture |
| Flow | Marshall flow value |
| ITSM 20°C | Indirect Tensile Stiffness Modulus measured at 20°C |
| ITSM 30°C | Indirect Tensile Stiffness Modulus measured at 30°C |

Training separate models allows each output variable to have its own fuzzy rule structure and consequent parameters.


## Input Features

The model uses 10 continuous input variables describing the asphalt mixture.

Before model training, the input variables are converted to numerical form and standardized.

Standardization is performed using statistics calculated from the training data:

`z = (x - mean) / std`

The same training mean and standard deviation are then applied to the test set and interactive inputs.


## Data Preprocessing

The preprocessing pipeline includes:

1. Loading the asphalt dataset
2. Selecting the required input and target columns
3. Converting values to numerical format
4. Handling invalid or missing observations
5. Splitting the dataset into training and testing sets
6. Standardizing the input features
7. Training a separate model for each target variable

The dataset is divided into:

- 80% training data
- 20% testing data

A fixed random seed of `42` is used to make the train/test split reproducible.


# Fuzzy C-Means Clustering

## Overview

Fuzzy C-Means is used to identify fuzzy regions in the input space.

Unlike hard clustering methods, FCM allows each observation to belong to multiple clusters with different membership degrees.

For each sample `i` and cluster `k`, the algorithm calculates a membership value:

`u_ik`

where higher values indicate stronger membership of the sample in that cluster.


## FCM Objective

The clustering process minimizes the weighted within-cluster distance:

`J = sum_i sum_k (u_ik^m) ||x_i - c_k||^2`

where:

- `u_ik` is the membership degree
- `m` is the fuzzifier
- `x_i` is a training sample
- `c_k` is the center of cluster `k`

The algorithm iteratively updates cluster centers and membership values until convergence.


## From Clusters to Fuzzy Rules

Each FCM cluster is transformed into one TSK fuzzy rule.

For every cluster:

- The cluster center defines the center of the antecedent membership functions
- The cluster spread determines the Gaussian width
- A local linear consequent model is estimated from the training samples

Therefore, the number of clusters directly determines the number of fuzzy rules.


# TSK Fuzzy Inference System

## Gaussian Antecedents

Gaussian membership functions are used for the antecedent part of each fuzzy rule.

For feature `j` and rule `k`, the membership degree can be represented as:

`mu_kj(x_j) = exp(-0.5 * ((x_j - c_kj) / sigma_kj)^2)`

where:

- `c_kj` is the center
- `sigma_kj` controls the spread of the fuzzy set


## Rule Firing Strength

The activation of each rule is determined by combining the membership values of all input variables.

Conceptually:

`w_k(x) = product_j mu_kj(x_j)`

The firing strengths are normalized before combining rule outputs:

`w_bar_k = w_k / sum_r(w_r)`


## Linear Consequents

Each TSK rule contains a first-order linear consequent:

`f_k(x) = b_k0 + b_k1*x1 + ... + b_kn*xn`

The final prediction is calculated as the weighted combination of the individual rule outputs:

`y_hat = sum_k w_bar_k * f_k(x)`


# Consequent Parameter Estimation

The parameters of each local linear model are estimated using Weighted Least Squares.

Training observations with stronger activation for a particular fuzzy rule have a greater influence on that rule's consequent parameters.

Ridge regularization is also applied to improve numerical stability and reduce problems caused by ill-conditioned matrices.

Conceptually, the regularized solution is:

`beta = (X^T W X + lambda*I)^(-1) X^T W y`

where:

- `X` is the design matrix
- `W` contains the rule-specific sample weights
- `lambda` is the ridge regularization coefficient
- `beta` contains the consequent parameters


# Model Selection

Different numbers of fuzzy rules are evaluated during training.

For every candidate configuration:

1. FCM clustering is performed on the training data.
2. Cluster information is converted into fuzzy antecedents.
3. TSK consequent parameters are estimated.
4. Predictions are generated.
5. Forecasting errors are calculated.
6. The configurations are compared.

The final configuration is selected according to predictive performance.


# Evaluation Metrics

Several regression metrics are used to evaluate the models.


## RMSE

Root Mean Squared Error measures the average magnitude of prediction errors while giving greater importance to large errors.

`RMSE = sqrt(mean((y - y_hat)^2))`


## MAE

Mean Absolute Error measures the average absolute prediction error.

`MAE = mean(|y - y_hat|)`


## MAPE

Mean Absolute Percentage Error expresses the prediction error relative to the actual values.

`MAPE = mean(|(y - y_hat) / y|) * 100`


## R²

The coefficient of determination measures how much of the variation in the target variable is explained by the model.

Values closer to `1` indicate a stronger fit.


# Results

The trained TSK models were evaluated on the held-out test data.

The reported test RMSE values are:

| Target | Test RMSE |
| --- | ---: |
| Stability | 1.2631 |
| Flow | 0.8741 |
| ITSM 20°C | 673.86 |
| ITSM 30°C | 254.27 |

The results show that the fuzzy-rule-based approach can model nonlinear relationships between asphalt mixture variables and the selected target properties.

Because the four target variables are measured on different numerical scales, their RMSE values should be interpreted independently rather than compared directly.


# Prediction Analysis

The final models combine the interpretability of fuzzy systems with the flexibility of local regression.

FCM divides the standardized feature space into overlapping fuzzy regions. Each region is represented by a separate TSK rule with its own linear consequent.

As a result, the overall nonlinear prediction problem is represented as a weighted combination of simpler local models.


# Visualizations

The project generates plots for evaluating the trained models.

These visualizations can be used to compare:

- Actual target values
- Predicted target values
- Prediction errors
- Model behavior across different target properties

If the generated figures are stored in an `outputs` directory, they can also be displayed directly in this README.


# Project Structure

```text
tsk-fuzzy-asphalt-prediction/
|
├── README.md
├── requirements.txt
├── main.py
├── preprocess.py
├── fcm.py
├── tsk.py
├── experiments.py
├── metrics.py
├── visualize.py
├── cli.py
├── asphalt_dataset.xlsx
└── outputs/
```

## File Description

| File | Description |
| --- | --- |
| `main.py` | Main entry point for running the complete modeling pipeline |
| `preprocess.py` | Data loading, cleaning, splitting, and standardization |
| `fcm.py` | Fuzzy C-Means clustering implementation |
| `tsk.py` | TSK fuzzy inference system and prediction logic |
| `experiments.py` | Training and comparison of different model configurations |
| `metrics.py` | Regression evaluation metrics |
| `visualize.py` | Visualization of model predictions and results |
| `cli.py` | Interactive prediction interface |
| `requirements.txt` | Required Python packages |
| `outputs/` | Generated results and visualizations |


# Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Enter the project directory:

```bash
cd tsk-fuzzy-asphalt-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```


# Usage

Run the main program:

```bash
python main.py
```

The program performs the main stages of the project, including preprocessing, fuzzy clustering, TSK model training, evaluation, and result generation.


## Interactive Prediction

The interactive interface can be used to enter new asphalt mixture measurements and obtain predictions from the trained models.

Run:

```bash
python cli.py
```

The entered values are processed using the same preprocessing parameters used during training before being passed to the fuzzy inference system.


# Technologies and Methods

- Python
- NumPy
- Pandas
- Matplotlib
- Fuzzy Logic
- Takagi-Sugeno-Kang Fuzzy Systems
- Fuzzy C-Means Clustering
- Gaussian Membership Functions
- Weighted Least Squares
- Ridge Regularization
- Regression
- Data Standardization
- Data Visualization


# Key Concepts Demonstrated

This project demonstrates several concepts in fuzzy systems and machine learning:

- Fuzzy clustering
- Soft membership
- Fuzzy IF-THEN rules
- TSK fuzzy inference
- Gaussian antecedent membership functions
- Local linear regression
- Weighted parameter estimation
- Rule-based nonlinear modeling
- Regression evaluation
- Data preprocessing
- Model selection


# Possible Improvements

Possible extensions of this project include:

- Testing additional numbers of fuzzy rules
- Comparing alternative membership functions
- Automatic optimization of fuzzy-system parameters
- Feature selection
- Cross-validation for model selection
- Comparison with conventional regression models
- Comparison with neural-network-based predictors
- Improved visualization of fuzzy rules
- Saving and loading trained models
- Development of a graphical prediction interface


# Course Information

**Course:** Fuzzy Sets and Systems  
**University:** Shiraz University


# Author

Saghar Kheradmand
