# Python + Streamlit Data Cleaning Agent

## Project Overview
This project is a Python-based Data Cleaning Agent designed for machine learning dataset preparation. It provides a guided workflow for uploading a dataset, validating it, profiling it, cleaning it, generating output files, and recommending a suitable machine learning problem type and starter algorithms.

The original academic guideline suggested using Flowise + n8n + Python. In this implementation, the orchestration and user interaction layer is built with Streamlit, while Python remains the actual data processing and cleaning engine. This keeps the project aligned with the guideline while using a Python-only architecture that was allowed for the midterm.

## Teacher Guideline Alignment
This implementation aligns with the teacher's required flow in the following ways:

- Accepts a dataset from the user through file upload
- Validates the dataset before any further processing
- Profiles the dataset to understand its structure and quality
- Runs Python automatically behind the interface
- Uses Pandas and NumPy for data handling
- Cleans the dataset based on selected options
- Generates a cleaned dataset file
- Generates a cleaning report file
- Shows explanations and summaries to the user through the Streamlit interface

The original guideline used Flowise + n8n + Python. This project uses Streamlit + Python instead, because a full Python implementation was permitted and Python already performs the actual cleaning logic.

## Features
- Upload CSV and Excel files
- Pre-cleaning validation
- Dataset profiling
- Missing value handling
- Duplicate row removal
- Wrong data type detection and fixing
- Outlier detection and capping using IQR
- Categorical encoding using one-hot encoding
- Numeric scaling using `StandardScaler` or `MinMaxScaler`
- NLP text cleaning
- ML problem type recommendation
- Algorithm recommendation
- Cleaned CSV generation
- JSON cleaning report generation
- Download buttons for output files
- Friendly error handling

## Technology Stack
- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- openpyxl

## Project Structure
```text
data-cleaning-agent/
|-- app.py
|-- requirements.txt
|-- README.md
|-- output/
|-- reports/
|-- utils/
|   |-- data_loader.py
|   |-- data_validator.py
|   |-- data_profiler.py
|   |-- data_cleaner.py
|   |-- nlp_cleaner.py
|   |-- ml_recommender.py
|   `-- report_generator.py
|-- test_validation.py
|-- test_five_dataset_complete.py
|-- test_full_dataset_workflows.py
|-- test_data_profiler.py
|-- test_profiling.py
|-- test_upload.py
|-- test_ui.py
`-- test_edge_case_safety.py
```

If you want to keep sample datasets locally, you can also add an `input/` folder for convenience.

## Application Execution Flow
The application follows this execution order:

Upload dataset  
→ Load dataset  
→ Validate dataset  
→ Profile dataset  
→ Select cleaning options  
→ Clean dataset  
→ Generate cleaned CSV  
→ Generate JSON report  
→ Recommend ML algorithm  
→ Download output files

Important behavior:

- Validation happens before profiling and cleaning.
- If validation fails, profiling and cleaning are blocked.
- This prevents the app from continuing with unusable or unsafe input data.

## Module Explanation
### `app.py`
Main Streamlit user interface and execution flow.

Responsibilities:
- Builds the sidebar inputs
- Accepts uploaded files
- Lets the user choose the target column
- Runs validation before profiling
- Shows the profiling report
- Shows ML recommendations
- Runs the cleaning pipeline
- Saves the cleaned CSV and JSON report
- Displays download buttons and cleaning summaries

### `utils/data_loader.py`
Loads CSV and Excel files into a Pandas `DataFrame`.

Responsibilities:
- Accepts `.csv`, `.xlsx`, and `.xls`
- Resets file pointers safely for repeated reads
- Handles invalid, empty, or unreadable uploads gracefully

### `utils/data_validator.py`
Checks whether the dataset is usable before profiling and cleaning.

Responsibilities:
- Verifies supported file type
- Detects missing or invalid DataFrame input
- Blocks empty datasets
- Detects duplicate or blank column names
- Checks whether the selected target column exists
- Produces warnings for risky but still usable datasets

### `utils/data_profiler.py`
Profiles the dataset and identifies the main data characteristics.

Responsibilities:
- Detects dataset shape
- Captures column names and data types
- Counts missing values
- Detects duplicate rows
- Classifies numeric, categorical, text, datetime, boolean, and ID-like columns
- Supplies reusable metadata for cleaning and ML recommendation

### `utils/data_cleaner.py`
Handles the main dataset cleaning tasks.

Responsibilities:
- Removes duplicate rows when selected
- Fills missing values
- Safely converts wrong data types when a conversion is reliable
- Detects and caps outliers using the IQR method
- Encodes categorical columns with one-hot encoding
- Scales numeric feature columns
- Protects the target column from unsafe transformations
- Applies optional NLP cleaning to detected text columns
- Returns a detailed cleaning summary

### `utils/nlp_cleaner.py`
Cleans text columns for beginner-friendly NLP preparation.

Responsibilities:
- Detects likely free-text columns
- Converts text to lowercase
- Removes URLs, emails, HTML, punctuation, and special symbols
- Removes extra spaces
- Supports stopword removal
- Keeps backup copies of original text columns
- Preserves the target label column from text cleaning

### `utils/ml_recommender.py`
Detects the likely ML problem type and recommends suitable starter algorithms.

Responsibilities:
- Handles no-target clustering scenarios
- Detects NLP/text classification scenarios
- Distinguishes classification from regression using target behavior
- Recommends beginner-friendly algorithms for each problem type

### `utils/report_generator.py`
Generates the JSON cleaning report and saves output files.

Responsibilities:
- Builds a report payload from profiling, validation, cleaning, and ML metadata
- Converts values into JSON-safe structures
- Saves the report into the `reports/` folder
- Generates safe filenames for outputs

## Installation
Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application
```bash
streamlit run app.py
```

## How to Use
1. Start the Streamlit app.
2. Upload a CSV or Excel dataset.
3. Select a target column if available.
4. Select a problem type or keep `Auto-detect`.
5. Choose the cleaning options you want to apply.
6. Click `Clean Dataset`.
7. Review the cleaning summary.
8. Review the ML recommendation.
9. Download the cleaned CSV and JSON report.

## Supported Dataset Types
- Classification datasets, for example Titanic
- Regression datasets, for example House Prices
- NLP/Text classification datasets, for example Spam SMS
- Clustering or no-target datasets, for example Customer Segmentation

## ML Recommendation Rules
The app uses simple and explainable rules to recommend a machine learning problem type:

- No target column selected → Clustering
- Text column plus target column → NLP/Text Classification
- Categorical target behavior → Classification
- Continuous numeric target behavior → Regression

Recommended beginner-friendly algorithms:

### Classification
- Logistic Regression
- Decision Tree
- Random Forest

### Regression
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### NLP/Text Classification
- Naive Bayes
- Logistic Regression
- Linear SVM

### Clustering
- K-Means
- DBSCAN

## Cleaning Logic Explanation
The cleaning pipeline is designed to be explainable and safe for academic and beginner use.

### Missing Values
- Missing numeric values are filled with the median.
- Missing non-numeric categorical-style values are filled with the mode when available.
- If no mode exists for a non-numeric column, the value `Unknown` is used.
- During NLP cleaning, text values are safely converted to strings and missing text is handled as empty text for preprocessing.

### Duplicate Rows
- Duplicate rows are removed only when the user selects that option.

### Wrong Data Types
- Object columns are converted only when they look reliably numeric or datetime-like.
- Unsafe conversions are skipped instead of forcing them.

### Outliers
- Outliers are capped using the IQR method.
- Extreme values are not blindly deleted.

### Categorical Encoding
- Categorical feature columns are encoded using `pd.get_dummies`.
- The target column is kept separate from feature encoding.

### Numeric Scaling
- Numeric columns are scaled only when selected by the user.
- The app supports `StandardScaler` and `MinMaxScaler`.
- The target column is protected from scaling.

### NLP Cleaning
- Text cleaning is applied only when selected.
- Text columns are lowercased and cleaned by removing URLs, emails, HTML, punctuation, special characters, numbers, and extra whitespace.
- Common English stopwords can be removed.
- Backup columns with `_original` suffix are kept for reference.
- The target label column is not altered by NLP cleaning.

## Output Files
The app generates two main output files:

### `output/cleaned_<filename>.csv`
This file contains the cleaned dataset after the selected transformations have been applied.

### `reports/cleaning_report_<filename>.json`
This file contains a structured JSON report with:

- Original dataset size
- Final dataset size
- Missing values before and after cleaning
- Duplicate row removal summary
- Column type classifications
- Encoding and scaling metadata
- Outlier handling details
- NLP cleaning summary
- Validation warnings or errors
- Recommended ML problem type and algorithms
- Output file paths

## Testing
Recommended test cases for demonstration and verification:

- Titanic dataset → Classification
- House Prices dataset → Regression
- Spam SMS dataset → NLP/Text Classification
- Customer Segmentation dataset with no target → Clustering
- Text dataset with no target → Clustering, not NLP classification

This repository also includes automated test files covering validation behavior, profiling, upload flow, UI behavior, workflow execution, and edge-case safety.

### Manual Flowise Greeting Test
Use this simple manual smoke test to confirm that the Flowise explanation layer is reachable even without dataset context.

Test input:

- Question: `Hey, what do you do?`

Expected result:

- Flowise API call succeeds
- Streamlit displays the returned answer
- No dataset summary is required for this test
- A valid response may look like:
  `I am a Data Cleaning Agent. I help analyze datasets and suggest cleaning steps for machine learning preparation.`

### Manual Flowise Titanic Classification Test
Use this manual test to verify that the Flowise explanation layer produces an exam-focused classification answer from the summarized Titanic dataset profile.

Test dataset columns:

- `PassengerId`
- `Survived`
- `Pclass`
- `Name`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Ticket`
- `Fare`
- `Cabin`
- `Embarked`

Target column:

- `Survived`

Question to Flowise:

- `Please give me a full 25/25 midterm answer covering Part A: Agent Development in Flowise - 10 Marks and Part B: Case Study Analysis - 15 Marks.`

Expected output should include:

- Dataset type: Tabular dataset
- ML problem type: Binary classification
- Target column: `Survived`
- Cleaning steps:
- Fill `Age` using median
- Fill `Embarked` using mode
- Drop `Cabin` or create `has_cabin`
- Remove duplicates
- Fix wrong data types
- Check outliers in `Age` and `Fare`
- Encode `Sex` and `Embarked`
- Scale `Age` and `Fare` if using Logistic Regression
- Recommended algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- Includes Pandas and NumPy usage
- Includes Part A and Part B headings
- The question text area should not contain dataset content
- The `Preview sent to AI Agent` expander should show only a compact Python-generated preview
- Only the preview text, not the full dataset, should be sent to Flowise

### Manual Flowise House Prices Regression Test
Use this manual test to verify that the Flowise explanation layer correctly identifies a regression problem and avoids treating the target as classification.

Test dataset columns:

- `Id`
- `LotArea`
- `OverallQual`
- `YearBuilt`
- `TotalBsmtSF`
- `GrLivArea`
- `BedroomAbvGr`
- `GarageCars`
- `GarageArea`
- `Neighborhood`
- `HouseStyle`
- `SalePrice`

Target column:

- `SalePrice`

Question to Flowise:

- `Analyze this dataset and recommend cleaning steps and suitable ML algorithms.`

Expected output should include:

- Dataset type: Tabular dataset
- ML problem type: Regression
- Target column: `SalePrice`
- It must not say classification
- It must not mention class imbalance as a main issue
- Cleaning steps:
- Fill numeric missing values with median
- Fill categorical missing values with mode or `Unknown`
- Encode `Neighborhood` and `HouseStyle`
- Check outliers in `LotArea`, `GrLivArea`, and `SalePrice`
- Consider `np.log1p` for skewed `SalePrice`
- Recommended algorithms:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### Manual Flowise Spam SMS NLP Test
Use this manual test to verify that the Flowise explanation layer correctly identifies an NLP text classification task and prioritizes text preprocessing over generic tabular scaling advice.

Test dataset columns:

- `label`
- `message`

Target column:

- `label`

Question to Flowise:

- `Analyze this NLP dataset and suggest cleaning steps and suitable ML algorithms.`

Expected output should include:

- Dataset type: NLP or text dataset
- ML problem type: Text classification
- Target column: `label`
- Text column: `message`
- Cleaning steps:
- Lowercase text
- Remove punctuation
- Tokenize text
- Remove stopwords
- Optional stemming or lemmatization
- Use TF-IDF or Bag-of-Words
- Encode `ham` and `spam` labels
- Use train/test split with `stratify=y`
- Fit TF-IDF only on `X_train`
- Recommended algorithms:
- Naive Bayes
- Logistic Regression
- Linear SVM
- It must not focus on `StandardScaler`, `MinMaxScaler`, IQR, or Z-score as the main NLP steps

### Manual Flowise No-Target Clustering Test
Use this manual test to verify that the Flowise explanation layer correctly identifies an unsupervised clustering problem when no target column is provided.

Test dataset columns:

- `CustomerID`
- `Age`
- `Income`
- `SpendingScore`
- `PurchaseFrequency`

Target column:

- `None`

Question to Flowise:

- `Analyze this dataset. No target column is provided. What is the correct ML problem type?`

Expected output should include:

- Problem type: Clustering or unsupervised learning
- Reason: no target column is provided and the goal is grouping similar records
- `CustomerID` should be removed or excluded as an identifier
- Missing numeric values should be filled with median
- Numeric features should be scaled
- Recommended algorithms:
- K-Means
- DBSCAN
- It should ask for a target column if the user wants classification or regression

## Edge Cases Handled
- No file uploaded
- Invalid CSV or unsupported file type
- Empty dataset
- Duplicate columns
- Dataset with only numeric columns
- Dataset with only text columns
- No target column selected
- No categorical columns
- No numeric columns
- Cleaning option selected but no matching columns found

## Limitations
- This app prepares data for machine learning but does not train models.
- Very large datasets may take more time to process.
- Automatic cleaning may still require human review for business-specific decisions.
- Advanced NLP steps such as lemmatization are intentionally kept simple to avoid extra downloads during demo use.
- The app is not a full replacement for expert data analysis, but it is useful for academic work and beginner ML preparation.

## Viva Explanation
“I built a Python + Streamlit Data Cleaning Agent that uploads a dataset, validates it, profiles it, cleans it, generates a cleaned CSV file, creates a JSON report, and recommends a suitable ML algorithm. The app follows the correct order: load, validate, profile, and clean. Python with Pandas and NumPy performs the actual data processing, while Streamlit provides the user interface.”

## Screenshots
Add screenshots here:

- Streamlit homepage
- Dataset upload
- Validation result
- Dataset profile
- Cleaning options
- Cleaning summary
- Download buttons
- Output folder
- JSON report
