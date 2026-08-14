
# TSK Fuzzy System for Asphalt Property Prediction

## Overview

This project implements a Takagi-Sugeno-Kang (TSK) fuzzy inference system for predicting asphalt mixture properties from laboratory mix-design features.

Four separate fuzzy models are trained to predict:

- Adjusted Stability
- Flow
- ITSM at 20°C
- ITSM at 30°C

The core fuzzy modeling pipeline is implemented from scratch using NumPy, including Fuzzy C-Means clustering, Gaussian membership functions, and first-order TSK inference.

## Main Features

- Fuzzy C-Means rule generation
- Gaussian membership functions
- First-order TSK fuzzy models
- Weighted least squares
- Input standardization
- Train/test RMSE evaluation
- Interactive prediction interface

## Technologies

Python · NumPy · Pandas · OpenPyXL · Fuzzy Systems

## Author

Saghar Kheradmand
