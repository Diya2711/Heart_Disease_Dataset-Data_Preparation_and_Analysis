# ❤️ Heart Disease Dataset – Data Preparation and Analysis

## 📌 Project Overview

This project is a Python-based data preparation and analysis application developed using Streamlit. It combines three data analysis tasks into a single interactive web application and also includes separate Python files for running each task independently.

The application helps users load and browse datasets, calculate summary statistics, identify missing values, compare imputation techniques, and download processed data.

## 🎯 Project Tasks

### Task 1: Load and Browse Dataset

**File:** `task1_load_and_browse.py`

* Upload CSV, TSV, TXT, and DATA datasets.
* Automatically detect delimiters and column headers.
* Display the number of rows and columns.
* Sort data in ascending or descending order.
* Browse data using pagination.

### Task 2: Summary Statistics

**File:** `task2_summary_statistics.py`

Calculate descriptive statistics using manually implemented Python functions.

* Count of available values.
* Number of missing values.
* Mean, median, and mode.
* Minimum and maximum values.
* Sample standard deviation.
* Display results in a table.

### Task 3: Missing Value Identification and Imputation

**File:** `task3_missing_values.py`

* Identify missing cells and affected rows.
* Calculate missing-value counts and percentages.
* Visualize missing values using a bar chart.
* Compare Mean/Mode, Median/Mode, Group-wise, and k-NN imputation.
* Evaluate methods using artificially hidden values.
* Select an imputation method.
* Preview and download the imputed dataset.
* Generate and download a report explaining the selected method.

## 🚀 Combined Streamlit Application

**File:** `app.py`

The `app.py` file integrates all three tasks into a single application called **Heart Disease – Data Preparation**.

Instead of running three separate applications, users can access all the features through one interface with three tabs.

### Main Features

* **Interactive dataset upload:** Upload a dataset through the sidebar or provide a local file path.
* **Dataset browsing:** View row and column counts, detect delimiters, sort records, and navigate pages.
* **Summary statistics:** Calculate descriptive statistics using manually implemented functions.
* **Missing-value analysis:** Identify missing values and visualize their distribution.
* **Flexible target selection:** Choose the label column from the sidebar.
* **Zero-value handling:** Optionally treat zero values in `trestbps`, `chol`, and `thalach` as missing.
* **Imputation comparison:** Evaluate four imputation methods and rank them by reconstruction error.
* **Data export:** Download the imputed dataset as a CSV file.
* **Report generation:** Download a Markdown report containing findings and method justification.

### Application Tabs

| Tab                    | Purpose                                                                   |
| ---------------------- | ------------------------------------------------------------------------- |
| 1 · Load & Browse      | Explore, sort, and paginate dataset records.                              |
| 2 · Summary Statistics | Calculate descriptive statistics for numerical columns.                   |
| 3 · Missing Values     | Identify missing data, compare imputation techniques, and export results. |

## 🛠️ Technologies Used

* Python
* Streamlit
* Random module
* Manual statistical calculations
* Data parsing and preprocessing

The application uses custom Python functions for its statistical calculations and imputation logic rather than relying on pandas for these operations.

## 📂 Project Structure

```text
Heart-Disease-Data-Preparation/
│
├── app.py
├── task1_load_and_browse.py
├── task2_summary_statistics.py
├── task3_missing_values.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

### 2. Open the Project Directory

```bash
cd YOUR-REPOSITORY
```

### 3. Install Dependencies

Create a `requirements.txt` file with the following content:

```text
streamlit
```

Install the required dependency:

```bash
pip install -r requirements.txt
```

Python must be installed before running the application.

## ▶️ Run the Combined Application

To launch all three tasks through one interface, run:

```bash
streamlit run app.py
```

The application will display a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open this URL in your browser.

## ▶️ Run Individual Tasks

You can also execute each task independently.

**Task 1:**

```bash
streamlit run task1_load_and_browse.py
```

**Task 2:**

```bash
streamlit run task2_summary_statistics.py
```

**Task 3:**

```bash
streamlit run task3_missing_values.py
```

## 📁 Dataset Requirements

The application supports `.csv`, `.tsv`, `.txt`, and `.data` files.

The code includes recognized column names for the UCI Heart Disease dataset:

`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, and `target`.

For the combined application, you can upload a dataset using the sidebar. Alternatively, place a compatible dataset named `heart.csv` in the same directory as `app.py`, or enter another valid local file path.

The application allows the target column to be selected through the sidebar.

## 📊 Imputation Evaluation

The application evaluates imputation methods by hiding a sample of known values, filling them using each method, and comparing the estimated values with the original values.

The methods are:

1. Mean / Mode
2. Median / Mode
3. Group-wise imputation
4. k-Nearest Neighbors (k-NN, k = 5)

Numerical features are evaluated using normalized reconstruction error, while categorical features use an error rate. Lower error indicates better reconstruction performance in this experiment.

The application keeps the target column out of the imputation process to avoid target leakage.

## 🎓 Learning Outcomes

This project demonstrates:

* Data loading and preprocessing.
* Automatic delimiter and header detection.
* Data exploration and pagination.
* Manual implementation of statistical functions.
* Missing-value identification and visualization.
* Comparison of data imputation methods.
* Evaluation of reconstruction errors.
* Interactive dashboard development with Streamlit.
* Exporting processed datasets and reports.

## ⚠️ Limitations

* The dataset should have a consistent tabular structure.
* Summary statistics are displayed for numerical columns.
* Imputation evaluation uses artificially hidden known values and cannot verify the true values of naturally missing entries.
* Results depend on the dataset and the imputation technique.
* When using imputed data in machine learning, fit the imputer on the training split only to reduce the risk of data leakage.

## 🔮 Future Improvements

* Add more statistical measures and visualizations.
* Support additional missing-value handling techniques.
* Add dataset filtering and search.
* Generate downloadable analysis summaries.
* Integrate a heart disease prediction model.
* Deploy the application online using Streamlit Community Cloud.


