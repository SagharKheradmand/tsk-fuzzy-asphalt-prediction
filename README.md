# TSK Fuzzy System for Asphalt Property Prediction

## Overview

This project implements a Takagi-Sugeno-Kang (TSK) fuzzy inference system for predicting several properties of asphalt mixtures from laboratory mix-design data.

The model combines Fuzzy C-Means (FCM) clustering with first-order TSK fuzzy rules. FCM is used to identify fuzzy regions in the input space, while each fuzzy rule contains a linear consequent model.

Four independent TSK models are trained to predict:

- Adjusted Stability
- Flow
- ITSM at 20°C
- ITSM at 30°C

The main fuzzy modeling components are implemented in Python using NumPy without relying on a ready-made fuzzy logic library.


## Main Features

- First-order Takagi-Sugeno-Kang fuzzy inference system
- Fuzzy C-Means clustering
- Gaussian antecedent membership functions
- Linear rule consequents
- Weighted Least Squares parameter estimation
- Ridge regularization
- Four independent regression models
- Input feature standardization
- Reproducible train/test splitting
- RMSE-based model evaluation
- Interactive command-line prediction
- Custom fuzzy modeling implementation using NumPy


## Problem Description

The objective of this project is to predict important mechanical properties of asphalt mixtures from laboratory measurements and mix-design characteristics.

The model receives a vector of 10 continuous input features and predicts one asphalt property at a time.

A first-order TSK fuzzy rule can be represented conceptually as:

`IF x1 is A1 AND x2 is A2 AND ... AND xn is An`

`THEN y = b0 + b1*x1 + b2*x2 + ... + bn*xn`

Each rule describes a local linear relationship between the input variables and the target property.

The final prediction is obtained by combining the outputs of all fuzzy rules according to their normalized firing strengths.


## Dataset

The dataset used in this project is stored in:

`data/Asphalt-Dataset-ToClass.xlsx`

It contains laboratory measurements related to asphalt binder characteristics, mixture composition, volumetric properties, aggregate gradation, and mechanical properties.

After preprocessing and removal of invalid rows, the dataset contains 168 valid numerical observations.

The resulting matrices contain:

- 168 samples
- 10 input features
- 4 target variables


## Input Features

The following 10 variables are used as model inputs:

| Feature | Description |
| --- | --- |
| `Viscosity (η) Pa.s` | Asphalt binder viscosity |
| `% A.C. by wt. (Pb)` | Asphalt cement content by total mixture weight |
| `% Eff. AC (Pbe)` | Effective asphalt content |
| `Max. Theo. (Gmm)` | Maximum theoretical specific gravity |
| `% Air Voids` | Percentage of air voids |
| `Unit Wt. (kg/m³)` | Compacted asphalt unit weight |
| `p200 (Percent Passing)` | Percentage passing the 0.075 mm sieve |
| `p4 (Cumulative Percent Retained)` | Aggregate gradation characteristic |
| `p38 (Cumulative Percent Retained)` | Aggregate gradation characteristic |
| `p34 (Cumulative Percent Retained)` | Aggregate gradation characteristic |


## Target Outputs

Four separate models are trained for the following target variables:

| Target | Description |
| --- | --- |
| `Adj. Stability kN` | Adjusted Marshall stability |
| `Flow (mm)` | Flow or deformation at maximum load |
| `ITSM (Mpa) - 20 deg` | Indirect Tensile Stiffness Modulus at 20°C |
| `ITSM (Mpa) - 30 deg` | Indirect Tensile Stiffness Modulus at 30°C |

Training separate models allows each target variable to have its own TSK consequent parameters.


# Data Preprocessing

The preprocessing stage prepares the raw Excel dataset before fuzzy model training.

The main steps are:

1. Load the Excel dataset.
2. Detect and handle the special header structure.
3. Resolve the configured feature and target column names.
4. Convert selected columns to numerical values.
5. Remove rows containing invalid, NaN, or infinite values.
6. Shuffle the cleaned observations.
7. Split the dataset into training and testing subsets.
8. Standardize the input features.

The dataset is divided into:

- 80% training data
- 20% testing data

Using the cleaned dataset, this results in:

- 134 training samples
- 34 testing samples

A random seed of `42` is used for reproducible shuffling and splitting.


## Input Standardization

The input features are standardized before clustering and model training.

The transformation is:

`z = (x - mean) / std`

The mean and standard deviation are calculated only from the training data.

The same parameters are then applied to the test data and to new inputs entered through the interactive interface.

For features with a standard deviation extremely close to zero, the standard deviation is replaced with `1.0` to avoid numerical instability.


# Fuzzy C-Means Clustering

## Overview

Fuzzy C-Means is used to divide the standardized input space into overlapping fuzzy regions.

Unlike hard clustering, where each sample belongs to exactly one cluster, FCM assigns a membership degree to every sample for every cluster.

Therefore, a sample may partially belong to several fuzzy regions at the same time.


## FCM Objective

The general objective of Fuzzy C-Means can be represented as:

`J = sum_i sum_k (u_ik^m) ||x_i - c_k||^2`

where:

- `u_ik` is the membership degree of sample `i` in cluster `k`
- `m` is the fuzziness parameter
- `x_i` is an input sample
- `c_k` is a cluster center

The cluster centers and membership values are updated iteratively until convergence or until the maximum number of iterations is reached.


## FCM Configuration

The current project uses the following Fuzzy C-Means parameters:

| Parameter | Value |
| --- | ---: |
| Number of clusters | 4 |
| Fuzziness parameter `m` | 2.2 |
| Maximum iterations | 100 |
| Convergence tolerance | `1e-5` |
| Random seed | 42 |

Because each FCM cluster is converted into a fuzzy rule, the model contains four TSK rules.


# TSK Fuzzy Inference System

## Rule Generation

The fuzzy regions generated by FCM are used to construct the antecedent part of the TSK rules.

Each cluster corresponds to one fuzzy rule.

Cluster centers determine the centers of the Gaussian antecedent functions, while cluster spreads are used to determine their widths.


## Gaussian Membership Functions

Gaussian functions are used to calculate the firing strength of each fuzzy rule.

Conceptually, the membership of an input vector in a fuzzy region can be written as:

`mu(x) = exp(-0.5 * sum(((x - c) / sigma)^2))`

where:

- `x` is the input vector
- `c` is the fuzzy cluster center
- `sigma` represents the cluster spread

Very small spread values are limited to a small positive value to improve numerical stability.


## Rule Firing Strength

For each input sample, the model calculates the firing strength of every fuzzy rule.

The firing strengths are then normalized so that their sum is approximately equal to one.

This allows several fuzzy rules to contribute simultaneously to the final prediction.


## Linear Consequents

Each TSK rule contains a first-order linear consequent.

For rule `r`, the consequent can be represented as:

`f_r(x) = b_r0 + b_r1*x1 + b_r2*x2 + ... + b_rn*xn`

where the coefficients are learned from the training data.


## Final Prediction

The output of the TSK system is calculated as the weighted combination of the outputs produced by the individual rules:

`y_hat = sum_r w_r * f_r(x)`

where `w_r` represents the normalized firing strength of rule `r`.

This structure allows the model to represent nonlinear relationships using a combination of local linear models.


# Consequent Parameter Estimation

The consequent parameters are estimated using Weighted Least Squares.

Samples that activate a fuzzy rule more strongly have a larger influence on the consequent parameters of that rule.

Ridge regularization is added to improve numerical stability.

The regularized estimation can be represented conceptually as:

`beta = (X^T W X + lambda*I)^(-1) X^T W y`

where:

- `X` is the design matrix
- `W` contains rule-specific weights
- `lambda` is the ridge regularization parameter
- `beta` contains the consequent coefficients

The project uses:

`lambda = 0.01`


# Training Process

Four independent TSK models are trained.

The training targets are:

1. Stability
2. Flow
3. ITSM at 20°C
4. ITSM at 30°C

For each target:

1. Fuzzy C-Means is applied to the training inputs.
2. Four fuzzy rules are generated.
3. Gaussian antecedent parameters are determined.
4. Rule firing strengths are calculated.
5. Linear consequent parameters are estimated.
6. Predictions are generated for the training data.
7. Predictions are generated for the testing data.
8. RMSE is calculated for both subsets.

Each target therefore has its own independently trained TSK model.


# Model Configuration

The current configuration used by the project is:

| Parameter | Value |
| --- | --- |
| Number of fuzzy rules | 4 |
| FCM fuzziness parameter | 2.2 |
| FCM maximum iterations | 100 |
| FCM tolerance | `1e-5` |
| Random seed | 42 |
| Ridge regularization | `1e-2` |
| Training ratio | 0.8 |
| Testing ratio | 0.2 |
| Input normalization | Enabled |
| Shuffle before split | Enabled |
| Split seed | 42 |


# Evaluation

The current implementation evaluates the models using Root Mean Squared Error (RMSE).

RMSE is defined as:

`RMSE = sqrt(mean((y_true - y_pred)^2))`

RMSE is calculated independently for every target on both the training and testing subsets.

The evaluation procedure therefore reports:

- Stability train RMSE
- Stability test RMSE
- Flow train RMSE
- Flow test RMSE
- ITSM 20°C train RMSE
- ITSM 20°C test RMSE
- ITSM 30°C train RMSE
- ITSM 30°C test RMSE


## Test Results

The reported test RMSE values are:

| Target | Test RMSE |
| --- | ---: |
| Stability | 1.2631 |
| Flow | 0.8741 |
| ITSM 20°C | 673.86 |
| ITSM 30°C | 254.27 |

The four target variables are measured on different numerical scales, so their RMSE values should be interpreted independently rather than directly compared with each other.


# Interactive Prediction

After model training and evaluation, the project starts an interactive command-line interface.

The interface asks the user to enter 10 input values:

```text
TSK Interface
Enter 10 input values separated by space.
Type 'exit' to quit.
```

The values should be entered in the following order:

```text
Viscosity, Pb, Pbe, Gmm, Air Voids, Unit Weight, p200, p4, p38, p34
```

An example input is:

```text
414 4.5 8.10667 2.493766 7.570949 2304.963855 7 40 13 0
```

The entered values are standardized using the same scaler fitted on the training data.

The four trained models then generate predictions for:

- Stability
- Flow
- ITSM at 20°C
- ITSM at 30°C

Entering `exit` closes the interface.


# Project Structure

```text
tsk-fuzzy-asphalt-prediction/
|
├── README.md
├── requirements.txt
├── main.py
├── config.py
├── data.py
├── preprocess.py
├── fcm.py
├── tsk.py
├── experiments.py
├── metrics.py
├── cli.py
└── data/
    └── Asphalt-Dataset-ToClass.xlsx
```


## File Description

| File | Description |
| --- | --- |
| `main.py` | Runs the complete data loading, preprocessing, training, evaluation, and prediction pipeline |
| `config.py` | Stores dataset, model, and experiment configuration parameters |
| `data.py` | Loads the Excel dataset and handles its header structure |
| `preprocess.py` | Extracts numerical data, removes invalid rows, splits the dataset, and standardizes inputs |
| `fcm.py` | Implements Fuzzy C-Means clustering |
| `tsk.py` | Implements Gaussian membership functions and the TSK fuzzy model |
| `experiments.py` | Trains four separate TSK models and evaluates their RMSE |
| `metrics.py` | Implements the RMSE evaluation metric |
| `cli.py` | Provides the interactive prediction interface |
| `requirements.txt` | Lists the required Python packages |
| `data/Asphalt-Dataset-ToClass.xlsx` | Asphalt mixture dataset used for model training and evaluation |


# Installation

Python 3 is required to run the project.

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Move into the project directory:

```bash
cd tsk-fuzzy-asphalt-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```


# Requirements

The project uses:

- NumPy
- Pandas
- OpenPyXL

These dependencies are listed in `requirements.txt`.


# Usage

Run the complete project with:

```bash
python main.py
```

On systems where Python is available through `python3`, use:

```bash
python3 main.py
```

Running `main.py` automatically performs the following steps:

1. Loads `data/Asphalt-Dataset-ToClass.xlsx`
2. Extracts the 10 input features and 4 targets
3. Cleans invalid observations
4. Creates the train/test split
5. Standardizes the input features
6. Trains four TSK fuzzy models
7. Calculates train and test RMSE
8. Prints the evaluation results
9. Starts the interactive prediction interface


# Technologies and Methods

- Python
- NumPy
- Pandas
- OpenPyXL
- Fuzzy Logic
- Takagi-Sugeno-Kang Fuzzy Systems
- Fuzzy C-Means Clustering
- Gaussian Membership Functions
- Weighted Least Squares
- Ridge Regularization
- Regression
- Feature Standardization


# Key Concepts Demonstrated

This project demonstrates several concepts in fuzzy systems and machine learning:

- Fuzzy clustering
- Soft cluster membership
- Fuzzy IF-THEN rules
- First-order TSK fuzzy inference
- Gaussian fuzzy regions
- Local linear modeling
- Weighted parameter estimation
- Ridge regularization
- Nonlinear regression through local models
- Data preprocessing
- Train/test evaluation
- Interactive inference


# Current Limitations

The current implementation focuses on the core fuzzy modeling pipeline.

At this stage:

- Models are retrained each time the program runs.
- Trained models are not saved to disk.
- The fitted scaler is not saved separately.
- No prediction plots are generated automatically.
- No result CSV or JSON files are saved.
- The number of fuzzy rules is fixed in the configuration.
- Evaluation currently uses RMSE only.


# Possible Improvements

Possible future extensions include:

- Saving trained models for later reuse
- Saving the fitted scaler
- Adding predicted-versus-actual plots
- Saving evaluation results to CSV or JSON
- Comparing different numbers of fuzzy rules
- Adding cross-validation
- Testing alternative membership functions
- Adding additional evaluation metrics
- Supporting command-line configuration
- Adding automated tests
- Adding batch prediction
- Comparing the TSK system with conventional regression and machine learning models


# Course Information

**Course:** Fuzzy Sets and Systems  
**University:** Shiraz University


# Author

Saghar Kheradmand
