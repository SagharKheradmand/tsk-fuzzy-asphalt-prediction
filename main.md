# TSK Fuzzy System for Asphalt Property Prediction

## Overview

This project implements a Takagi-Sugeno-Kang (TSK) fuzzy inference system for predicting asphalt mixture properties from laboratory mix-design features.

The system uses 10 continuous input features related to asphalt binder behavior, asphalt content, volumetric properties, density, and aggregate gradation. It trains four separate first-order TSK fuzzy models to predict:

- Adjusted Stability in kN
- Flow in mm
- ITSM at 20 deg C
- ITSM at 30 deg C

This is a machine learning / fuzzy systems project implemented in Python. The core fuzzy modeling logic is implemented from scratch using NumPy, including Fuzzy C-Means rule generation, Gaussian membership functions, weighted least squares consequent estimation, prediction, and RMSE evaluation.

## Key Features

- Loads an Excel asphalt dataset and handles the dataset's two-row header format.
- Cleans data by converting selected columns to numeric values and removing invalid rows.
- Uses an 80/20 train/test split with deterministic shuffling.
- Standardizes input features before clustering and model training.
- Generates fuzzy rules with Fuzzy C-Means clustering.
- Builds first-order TSK fuzzy rules with Gaussian antecedents and linear consequents.
- Trains four independent models, one for each target output.
- Reports train and test RMSE for each output.
- Provides an interactive command-line interface for entering new asphalt feature values and receiving predictions.

## Project Highlights

- Built an end-to-end fuzzy modeling pipeline for asphalt strength, deformation, and stiffness prediction.
- Implemented Fuzzy C-Means clustering and TSK fuzzy inference without using a ready-made fuzzy logic library.
- Applied data cleaning, feature scaling, train/test splitting, model training, evaluation, and interactive prediction.
- Organized the repository with dataset, documentation, archive, dependency, and GitHub-ready README files.

## Dataset

The dataset is a local/custom Excel dataset included in this repository:

```text
data/Asphalt-Dataset-ToClass.xlsx
```

The project files do not specify an external public dataset source or official download URL. The dataset description is included in:

```text
docs/Prj2-dataset description.pdf
```

### Dataset Structure

- File type: Excel workbook (`.xlsx`)
- Sheet used by the code: `Sheet1` / sheet index `0`
- Raw loaded shape after header correction: 169 rows x 14 columns
- Valid numeric rows used by the model: 168 rows
- Input matrix shape: 168 rows x 10 features
- Output matrix shape: 168 rows x 4 targets
- Train/test split: 134 training rows and 34 testing rows

The workbook contains an initial `INPUT` / `OUTPUT` grouping row. The loader detects this format and rereads the workbook using the second row as the header. The first remaining row contains text descriptions, so preprocessing converts selected columns to numeric values and removes that non-numeric row.

### Input Features

| Column | Description |
| --- | --- |
| `Viscosity (n) Pa.s` | Asphalt binder viscosity |
| `% A.C. by wt. (Pb)` | Asphalt cement content by total mix weight |
| `% Eff. AC (Pbe)` | Effective asphalt content |
| `Max. Theo. (Gmm)` | Maximum theoretical specific gravity |
| `% Air Voids` | Air void percentage |
| `Unit Wt. (kg/m3)` | Compacted asphalt unit weight |
| `p200 (Percent Passing)` | Percent passing the 0.075 mm sieve |
| `p4 (Cumulative Percent Retained)` | Coarse aggregate content indicator |
| `p38 (Cumulative Percent Retained)` | Coarse aggregate skeleton indicator |
| `p34 (Cumulative Percent Retained)` | Largest aggregate size fraction indicator |

### Target Outputs

| Column | Meaning |
| --- | --- |
| `Adj.  Stability kN` | Adjusted Marshall stability |
| `Flow   (mm)` | Deformation at maximum load |
| `ITSM (Mpa) - 20 deg` | Indirect tensile stiffness modulus at 20 deg C |
| `ITSM (Mpa) - 30 deg` | Indirect tensile stiffness modulus at 30 deg C |

### Preprocessing

- Resolves configured column names using tolerant normalized matching.
- Converts feature and target columns to numeric values with `pandas.to_numeric`.
- Removes rows containing `NaN`, infinite, or otherwise invalid numeric values.
- Shuffles the dataset using seed `42`.
- Splits the dataset into 80% training and 20% testing.
- Standardizes input features using `(X - mean) / std`.
- Protects against near-zero standard deviations by replacing them with `1.0`.

No additional feature engineering, encoding, augmentation, or missing-value imputation is specified in the current project files.

## Project Structure

```text
Project2/
|-- archives/
|   `-- Mehdi_Mortazavian_40435074.zip
|-- data/
|   `-- Asphalt-Dataset-ToClass.xlsx
|-- docs/
|   |-- Fuzzy-Project2-Doc.pdf
|   `-- Prj2-dataset description.pdf
|-- cli.py
|-- config.py
|-- data.py
|-- experiments.py
|-- fcm.py
|-- main.py
|-- metrics.py
|-- preprocess.py
|-- requirements.txt
|-- tsk.py
|-- .gitignore
`-- README.md
```

| Path | Purpose |
| --- | --- |
| `main.py` | Runs the full pipeline: load data, preprocess, train models, evaluate, and start the CLI. |
| `config.py` | Stores dataset, model, and experiment configuration values. |
| `data.py` | Loads Excel or CSV data and handles the dataset's special header format. |
| `preprocess.py` | Extracts input/output arrays, cleans invalid rows, splits data, and standardizes features. |
| `fcm.py` | Implements Fuzzy C-Means clustering from scratch. |
| `tsk.py` | Implements Gaussian membership functions and the TSK model. |
| `experiments.py` | Trains four independent TSK models and computes train/test RMSE. |
| `metrics.py` | Provides the RMSE metric. |
| `cli.py` | Provides an interactive prediction interface after training. |
| `data/` | Contains the local Excel dataset. |
| `docs/` | Contains the project report and dataset description PDFs. |
| `archives/` | Contains the original packaged project archive. |

## Methodology / Workflow

1. Dataset loading
   - `main.py` creates a `DataConfig` and loads `data/Asphalt-Dataset-ToClass.xlsx`.
   - `data.py` reads the workbook with pandas and corrects the header if the first row only contains `INPUT` / `OUTPUT` section labels.

2. Feature and target extraction
   - `preprocess.py` resolves the configured feature and target columns.
   - Selected columns are converted to numeric arrays.
   - Invalid rows are removed.

3. Train/test split
   - The cleaned dataset is shuffled with seed `42`.
   - 80% of rows are used for training and 20% for testing.

4. Normalization
   - Input features are standardized using a custom `StandardScaler`.
   - The scaler is fit on the training data and applied to both training and testing data.

5. Fuzzy rule generation
   - `fcm.py` applies Fuzzy C-Means to the training inputs.
   - Each cluster becomes one fuzzy rule.
   - Cluster centers and weighted standard deviations define Gaussian membership functions.

6. TSK model training
   - `tsk.py` trains a first-order TSK model.
   - Rule consequents are linear functions of the inputs.
   - Consequent parameters are estimated with weighted least squares and ridge regularization.

7. Evaluation
   - `experiments.py` trains four separate models for stability, flow, ITSM20, and ITSM30.
   - RMSE is calculated separately on the training and testing splits.

8. Inference
   - After training, `cli.py` allows the user to enter 10 input values.
   - The same scaler is applied to the input.
   - The four trained models output predicted values.

### Model Configuration

The current configuration in `config.py` is:

| Parameter | Value |
| --- | --- |
| Number of rules / clusters | `4` |
| FCM fuzziness parameter `m` | `2.2` |
| FCM maximum iterations | `100` |
| FCM tolerance | `1e-5` |
| Random seed | `42` |
| Ridge regularization lambda | `1e-2` |
| Train ratio | `0.8` |
| Normalize inputs | `True` |
| Shuffle before split | `True` |

## Visual Results

No saved visual outputs were found in the current project files.

The current source code does not save plots, charts, confusion matrices, or training curves. If visual documentation is needed, a useful future improvement would be to save RMSE comparison charts or predicted-vs-actual plots into an `outputs/` or `figures/` directory and reference them here.

## Installation

Python 3 is required. The project was verified with the existing local Python 3.13 virtual environment.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the full training, evaluation, and interactive prediction pipeline:

```bash
python3 main.py
```

If your shell does not map `python3` to the virtual environment after activation, run:

```bash
.venv/bin/python main.py
```

The script prints RMSE results and then starts an interactive prompt:

```text
TSK Interface
Enter 10 input values separated by space.
Type 'exit' to quit.
```

Enter feature values in this order:

```text
Viscosity, Pb, Pbe, Gmm, Air Voids, Unit Weight, p200, p4, p38, p34
```

Example input format:

```text
414 4.5 8.10667 2.493766 7.570949 2304.963855 7 40 13 0
```

Type `exit` to close the CLI.

## Training / Running the Project

Training is performed automatically when `main.py` is executed. The training process:

- Loads the Excel dataset from `data/Asphalt-Dataset-ToClass.xlsx`.
- Extracts 10 input features and 4 target outputs.
- Splits the cleaned data into train/test sets.
- Fits the input scaler on the training split.
- Trains four separate TSK fuzzy models.
- Prints train/test RMSE for each target.
- Starts the interactive prediction interface.

There are no command-line arguments in the current project files. Model and experiment parameters can be edited in `config.py`.

## Evaluation

The project uses Root Mean Squared Error (RMSE):

```text
RMSE = sqrt(mean((y_true - y_pred)^2))
```

RMSE is computed separately for each target on both training and testing data.

## Results

The following results were produced by running:

```bash
printf 'exit\n' | .venv/bin/python main.py
```

| Output | Train RMSE | Test RMSE |
| --- | ---: | ---: |
| Stability | 0.8855170065 | 1.2630586142 |
| Flow | 0.6215925702 | 0.8740897194 |
| ITSM 20 deg C | 480.1762957982 | 673.8586223534 |
| ITSM 30 deg C | 145.3206402905 | 254.2657660171 |

No trained model checkpoints, serialized scalers, plots, or log files are saved by the current code.

## Requirements

The dependencies are listed in `requirements.txt`:

```text
numpy
pandas
openpyxl
```

## Technologies Used

- Python
- NumPy
- Pandas
- OpenPyXL
- Excel workbook data
- Custom Fuzzy C-Means implementation
- Custom first-order TSK fuzzy inference implementation

## Future Improvements

- Save trained models and the fitted scaler for reuse without retraining.
- Add predicted-vs-actual plots and RMSE comparison charts under `outputs/` or `figures/`.
- Add command-line arguments for dataset path, number of rules, train ratio, and random seed.
- Add automated tests for preprocessing, FCM convergence behavior, TSK prediction shape, and RMSE calculation.
- Add cross-validation or repeated random splits for more robust evaluation.
- Save experiment results to JSON or CSV for easier comparison.
- Add a non-interactive prediction mode for batch inference.
- Add license metadata before publishing the repository.

## References

- Local dataset: `data/Asphalt-Dataset-ToClass.xlsx`
- Dataset description: `docs/Prj2-dataset description.pdf`
- Project report: `docs/Fuzzy-Project2-Doc.pdf`
- NumPy documentation: https://numpy.org/doc/
- Pandas documentation: https://pandas.pydata.org/docs/
- OpenPyXL documentation: https://openpyxl.readthedocs.io/

## License

No license file is currently included in this repository. Add a license before publishing if you want to define usage permissions.
