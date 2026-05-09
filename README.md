# Data Cleaning Agent for ML Dataset Preparation

## Overview
This project is a Python + Streamlit data cleaning app for machine learning dataset preparation. It lets a user upload a dataset, validate it, profile it, visualize it, apply beginner-friendly cleaning steps, generate a cleaned CSV, generate a JSON report, and optionally ask a Flowise-powered AI explanation layer for a human-readable explanation.

The important architecture decision is that **Python remains responsible for the real data work**:
- file upload and parsing
- validation
- profiling
- visualizations
- cleaning
- report generation
- cleaned CSV output

Flowise is used only as an **AI explanation layer** on top of a compact Python-generated dataset profile.

## Why This Project Exists
The app is designed for:
- academic demos and viva presentations
- beginner-friendly ML dataset preparation
- explainable preprocessing
- safer AI-assisted explanation without sending full raw datasets to an LLM

## Core Architecture
The project has five main layers:

1. **Streamlit UI**
- Handles file upload
- Shows dataset preview, metrics, charts, cleaning options, results, and downloads

2. **Python Data Pipeline**
- Loads the dataset
- Validates it before continuing
- Profiles data types and quality
- Applies actual cleaning transformations

3. **Pandas / NumPy Processing**
- Used for profiling, validation, cleaning, transformation summaries, and dataset statistics

4. **Flowise Explanation Layer**
- Receives a compact Python-generated dataset profile
- Returns a human-readable explanation
- Never owns the actual cleaning logic

5. **Output Layer**
- Cleaned CSV file
- JSON cleaning report
- Viva-ready project summary

## High-Level Workflow
The application follows this order:

`Upload -> Load -> Validate -> Profile -> Visualize -> Detect Issues -> Recommend ML Problem Type -> Clean -> Generate Report -> AI Explanation -> Download`

This order matters:
- validation happens before profiling and cleaning
- profiling happens before any transformation
- Flowise is called only with a compact profile, not the full raw file
- downloads remain available even if Flowise fails

## Main Features
- CSV and Excel upload support
- Pre-cleaning validation
- Dataset profiling
- Dataset visualizations
- Missing-value handling
- Duplicate-row removal
- Safe data-type fixing
- Outlier handling using the IQR method
- One-hot encoding for categorical features
- Numeric scaling with `StandardScaler` or `MinMaxScaler`
- Optional NLP text cleaning
- ML problem-type recommendation
- Beginner-friendly algorithm recommendation
- Pandas and NumPy usage summary
- Flowise AI explanation based on a compact dataset profile
- Viva-ready project summary
- Cleaned CSV output
- JSON report output
- Optional markdown downloads

## Current UI Sections
The Streamlit app currently presents the workflow in this order:

1. `Dataset Preview`
2. `Pre-cleaning Validation`
3. `Dataset Profile`
4. `Dataset Visualizations`
5. `Data Quality Report`
6. `Recommended ML Algorithm`
7. `Cleaning Options`
8. `AI Agent Explanation`
9. `Cleaning Summary`
10. `Before vs After Cleaning`
11. `Pandas and NumPy Usage`
12. `Project Summary for Viva`
13. `Download Output Files`

## Technology Stack
- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- openpyxl
- requests
- optional NLTK fallback support in NLP cleaning

## Project Structure
```text
Data-Cleaning-Agent/
├── app.py
├── README.md
├── requirements.txt
├── output/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── src/
│   └── ai/
│       └── flowise_client.py
├── utils/
│   ├── data_cleaner.py
│   ├── data_loader.py
│   ├── data_profiler.py
│   ├── data_validator.py
│   ├── library_usage.py
│   ├── ml_recommender.py
│   ├── nlp_cleaner.py
│   ├── report_generator.py
│   └── viva_summary.py
├── qa_tests/
│   ├── datasets/
│   ├── qa_runner.py
│   ├── QA_REPORT.md
│   └── qa_summary.json
├── test_before_after_cleaning_summary.py
├── test_cleaning_report_actions.py
├── test_data_profiler.py
├── test_dataset_profile_ui.py
├── test_edge_case_safety.py
├── test_five_dataset_complete.py
├── test_flowise_client.py
├── test_flowise_ui.py
├── test_full_dataset_workflows.py
├── test_ml_recommender_intelligence.py
├── test_profiling.py
├── test_ui.py
├── test_upload.py
└── test_validation.py
```

## Module-by-Module Explanation

### `app.py`
Main Streamlit application.

Responsibilities:
- build sidebar controls
- accept uploaded files
- let the user choose a target column
- render previews, metrics, charts, and summaries
- call validation, profiling, recommendation, cleaning, reporting, and Flowise explanation helpers
- render download buttons

### `utils/data_loader.py`
Loads CSV and Excel files into Pandas DataFrames.

Responsibilities:
- support `.csv`, `.xlsx`, `.xls`
- reset uploaded-file pointer safely
- handle invalid, empty, or unreadable files gracefully

### `utils/data_validator.py`
Performs pre-cleaning validation.

Responsibilities:
- check supported file type
- reject missing DataFrame input
- block empty datasets
- detect blank or duplicate column names
- verify selected target existence
- issue warnings for high missing values, all-missing columns, and high-cardinality text fields

### `utils/data_profiler.py`
Profiles the dataset before cleaning.

Responsibilities:
- dataset shape
- column names
- data types
- missing-value counts
- duplicate-row counts
- boolean, datetime, numeric-like, categorical, text, and ID-like classification
- `DataFrame.info()` capture for reporting

### `utils/data_cleaner.py`
Runs selected cleaning actions.

Responsibilities:
- duplicate removal
- missing-value filling
- safe numeric/datetime conversion
- outlier capping via IQR
- one-hot encoding
- numeric scaling
- NLP text cleaning
- before-vs-after metrics
- performed/skipped-step reporting

### `utils/nlp_cleaner.py`
Lightweight text cleaning helpers.

Responsibilities:
- detect likely free-text columns
- lowercase text
- remove URLs, emails, HTML, punctuation, special symbols, and numbers
- remove stopwords
- keep original backup columns

### `utils/ml_recommender.py`
Infers likely ML problem type and recommends starter algorithms.

Responsibilities:
- suggest likely target columns
- distinguish classification, regression, NLP/text classification, clustering, and unknown cases
- create beginner-friendly algorithm recommendation summaries

### `utils/library_usage.py`
Builds the Pandas/NumPy usage summary.

Responsibilities:
- explain only the functions relevant to actions actually performed
- produce beginner-friendly text for the UI and JSON report

### `utils/viva_summary.py`
Builds the viva-ready project summary.

Responsibilities:
- summarize architecture
- summarize workflow
- summarize dataset-specific results
- explain why the compact-profile approach is better than sending the full raw file to an LLM

### `utils/report_generator.py`
Builds and writes the JSON cleaning report.

Responsibilities:
- combine validation, profiling, cleaning, recommendation, Flowise, and viva-summary metadata
- make values JSON-safe
- save reports under `reports/`

### `src/ai/flowise_client.py`
Handles the Flowise explanation layer.

Responsibilities:
- build a compact dataset profile
- validate profile completeness before call
- cap preview rows at 10
- truncate long sample text values
- embed the compact profile into the Flowise `question` payload
- avoid sending the full dataset
- safely handle API errors
- record Flowise integration metadata

## Cleaning Logic

### Duplicate Removal
- Uses `drop_duplicates()` when selected
- Skips safely if the user does not enable it

### Missing Values
- Numeric columns use the median
- Non-numeric columns use the mode
- If no mode exists, `Unknown` is used

### Wrong Data Types
- Object columns are converted only when they reliably look numeric or datetime-like
- Unsafe conversions are skipped

### Outlier Handling
- Uses the IQR method
- Caps extreme values instead of blindly deleting rows

### Categorical Encoding
- Uses `pd.get_dummies`
- Keeps the target column separate from feature encoding

### Numeric Scaling
- Supports `StandardScaler`
- Supports `MinMaxScaler`
- Protects the target column from scaling

### NLP Text Cleaning
- applied only when selected
- target label column is protected
- original text backup columns are preserved

## ML Recommendation Logic
The app uses explainable heuristics:

- no confident target and no supervised signal -> clustering
- meaningful text feature plus label-like target -> NLP/Text Classification
- binary or low-cardinality target -> classification
- continuous numeric target -> regression
- weak signal -> unknown / needs more information

Starter algorithms by problem type:

### Classification
- Logistic Regression
- Decision Tree
- Random Forest

### Regression
- Linear Regression
- Random Forest Regressor

### NLP/Text Classification
- Naive Bayes
- Logistic Regression
- Linear SVM

### Clustering
- K-Means
- DBSCAN

## Dataset Visualizations
The app includes safe pre-cleaning visualizations:
- missing-values bar chart
- target/class distribution chart
- boxplot for numeric columns
- correlation heatmap for numeric columns

If a chart is not applicable, the app shows a friendly skip message instead of crashing.

## Flowise Integration Design
Flowise is intentionally limited to explanation only.

### What Python Sends
Python builds a compact profile object that includes:
- dataset name
- original file name
- rows and columns
- column names
- dtypes
- numeric/categorical/text/datetime columns
- missing-value summary
- total missing values
- duplicate count
- sample rows capped at 10
- cleaning actions performed
- skipped cleaning steps
- before-vs-after summary
- problem type and reason
- selected and suggested target column
- recommended algorithms with reasons
- Pandas/NumPy usage summary

### What Python Does Not Send
- full raw CSV
- full uploaded file content
- complete column value lists
- all rows of the dataset

### Why This Design Is Better
- avoids token limit errors
- reduces hallucination
- keeps exact calculations inside Python
- scales better for large datasets
- makes the Flowise explanation more controlled and auditable

### Flowise Failure Handling
If Flowise fails:
- the main app does not crash
- cleaned CSV download still works
- JSON report generation still works
- a friendly fallback message is shown
- Flowise error metadata is stored safely

## Output Files

### Cleaned CSV
`output/cleaned_<filename>.csv`

Contains the transformed dataset after selected cleaning actions.

### JSON Cleaning Report
`reports/cleaning_report_<filename>.json`

Contains:
- original and final dataset size
- missing values before and after cleaning
- duplicate-row removal summary
- profiling metadata
- validation warnings and errors
- encoding/scaling/outlier metadata
- NLP cleaning metadata
- ML recommendation
- algorithm recommendation
- Pandas/NumPy usage
- Flowise integration metadata
- project summary for viva
- output file paths

### Optional Markdown Downloads in UI
- AI explanation markdown
- viva summary markdown

## Project Summary for Viva
The app generates a short viva-ready summary that explains:
- architecture
- workflow
- current dataset
- problem type
- target column
- cleaning actions performed
- skipped steps
- recommended algorithm
- why the Python + compact-profile approach is safer than sending the full file to an LLM

## Installation

### 1. Create a Virtual Environment
```bash
python -m venv .venv
```

### 2. Activate It
macOS / Linux:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Run the App
```bash
streamlit run app.py
```

## How to Use
1. Start the app.
2. Upload a CSV or Excel dataset.
3. Optionally select a target column.
4. Review validation results.
5. Review the dataset profile and visualizations.
6. Review the ML problem-type and algorithm recommendation.
7. Choose cleaning options.
8. Optionally ask the Flowise AI explanation layer a question.
9. Click `Clean Dataset`.
10. Review the cleaning summary, Pandas/NumPy usage, and viva summary.
11. Download the cleaned CSV, JSON report, and optional markdown summaries.

## Supported Dataset Scenarios
- Titanic-style tabular classification
- House Prices-style tabular regression
- Spam SMS-style NLP/text classification
- Customer Segmentation-style no-target clustering
- text-only or numeric-only edge cases
- large CSV datasets with token-safe Flowise previews

## Testing
This repository includes both unit/UI tests and QA artifacts.

### Automated Test Coverage
- upload behavior
- validation behavior
- profiling behavior
- cleaning summaries
- ML recommendation logic
- Flowise payload generation
- visualization UI behavior
- edge-case safety
- full workflow checks

Test files include:
- `test_upload.py`
- `test_validation.py`
- `test_profiling.py`
- `test_data_profiler.py`
- `test_dataset_profile_ui.py`
- `test_cleaning_report_actions.py`
- `test_before_after_cleaning_summary.py`
- `test_ml_recommender_intelligence.py`
- `test_flowise_client.py`
- `test_flowise_ui.py`
- `test_visualizations_ui.py`
- `test_full_dataset_workflows.py`
- `test_five_dataset_complete.py`
- `test_edge_case_safety.py`
- `test_ui.py`

### QA Test Suite
The `qa_tests/` folder contains:
- `qa_runner.py`
- `QA_REPORT.md`
- `qa_summary.json`

The checked QA scenarios currently cover:
- Titanic classification
- House Prices regression
- Spam SMS NLP classification
- Customer segmentation without target
- no-missing-values dataset
- duplicate-row dataset
- no-numeric-columns dataset
- invalid CSV behavior
- large CSV behavior

## QA Note
The QA dataset CSVs in `qa_tests/datasets/` are now ignored by Git through `.gitignore`, so local QA files can be kept without polluting version control.

## Flowise Payload Summary
The app sends Flowise:
- a compact JSON-structured dataset profile
- embedded into the `question` payload
- plus up to 10 sample rows

The app does not send:
- the full uploaded CSV
- all records
- full column contents

## Limitations
- The app prepares data for ML but does not train models.
- Recommendations are heuristic and beginner-oriented.
- Some domain-specific cleaning decisions still require human review.
- Very large datasets may take longer to process in Python.
- NLP preprocessing is intentionally lightweight for demo and classroom use.

## Recommended Screenshots for Submission
- homepage
- uploaded dataset preview
- validation section
- dataset profile
- dataset visualizations
- algorithm recommendation
- Flowise preview payload
- cleaning summary
- Pandas/NumPy usage
- Project Summary for Viva
- download section
- generated JSON report

## Short Viva Explanation
You can explain the project like this:

> I built a Python + Streamlit Data Cleaning Agent for machine learning dataset preparation. Streamlit provides the interface, but Python performs the real validation, profiling, cleaning, and report generation using Pandas and NumPy. The app can clean tabular and text datasets, recommend a suitable ML problem type and starter algorithms, generate a cleaned CSV and JSON report, and send only a compact dataset profile to Flowise for a human-readable explanation. This approach is safer than sending the full raw file to an LLM because it avoids token overflow, reduces hallucination, and keeps exact calculations inside Python.
