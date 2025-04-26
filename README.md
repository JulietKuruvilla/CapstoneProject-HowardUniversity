# CapstoneProject-HowardUniversity
Exploring The Impact of Family History and Lifestyle Factors on Colorectal Cancer Risk
---
## Overview

This project aim to analyze and identify potential risk factors associated with colorectal cancer (CRC). By analyzing the potential risk factors, we aim to evaluate the independent and combined effects of identified risk factors on the incidence and progression of colorectal cancer. This project focuses on developing predictive models and identify which model works best to determine an individual’s risk factors for colorectal cancer. 

## Project Specific Key Features

- Integrates colorectal cancer, and lifestyle datasets.
- Develops predictive models using supervised machine learning algorithms.
- Identifies potential features at high risk contributing to the onset and progression of colorectal cancer.
- Aims to support data-driven decisions to determine an individual's risk factors for colorectal cancer.

## Getting Started

### Data Sources

We utilized the following datasets after submittig a request with the National Cancer Institute (NCI).

- **Pancreatic, Lung, Colorectal, and Ovarian (PLCO) Dataset from the National Cancer Institute (NCI):** The colorectal cancer dataset is a comprehensive dataset that contains all the PLCO study data available for colorectal cancer screening, incidence, and mortality analysis. This dataset contains one record for each of the approximately 155,000 participants in the PLCO trial. ([https://cdas.cancer.gov/datasets/plco/](https://cdas.cancer.gov/datasets/plco/))
- **Healthy Lifestyle Dataset from the National Cancer Institute (NCI):** The lifestyle dataset includes scores calculated for ~60,000 participants at baseline. ([https://cdas.cancer.gov/datasets/plco/](https://cdas.cancer.gov/datasets/plco/))

### Installation

- Clone the repository:
    ```bash
    git clone [repository URL]
    ```
- Ensure you have the necessary Python libraries installed:
    ```bash
    import numpy as np                  # Scientific Computing
    import pandas as pd                 # Data Analysis
    import matplotlib.pyplot as plt     # Plotting
    import seaborn as sns               # Statistical Data Visualization
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
    from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
    import category_encoders as ce
    import statsmodels.api as sm
    ```
- Download the datasets mentioned in the "Data Sources" section and ensure the file paths in the code match their locations.

## Utilization

This snippets show the initial steps in data exploration, cleaning, preprocessing, feature engineering, and model development.

- **Data Loading:** The Python code shows how to load the colorectal cancer, and lifestyle datasets using pandas. D
- **Data Cleaning and Preprocessing:** Datasets were merged using the primary key plco_id feature. The python code will show how to merge the datasets into a new CSV file as well. Missing values were handled, and feature selection was performed to focus on relevant variables for the health index score. 
- **Feature Engineering:** Filling missing values using Probabilistic Imputation, Feature revelance check using Cross Tabulation, Feature relationship check using Collinearilty and Correlation, and features selected using Chi-Square and Stat Models.
- **Data Transformation:** Applied OneHotEncoder and converted selected nominal categorical variables into a numerical format that can be provided to machine learning algorithms. Applied OrdinalEncoder to the ordinal features based on the defined scores. 
- **Exploratory Data Analysis:** Visualizations were generated to understand the relationships between variables.
- **Model Development:** The following eight supervised machine learning algorithms are employed to construct predictive models: Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Tree (DT), Random Forest (RF), Gradient Boosting (GB), AdaBoost, and XGBoost.
- **Model Training:** The data was split into training 80% and test 20% sets, and features were scaled using `StandardScaler`.
- **Model Evaluation:** Performance metrics such as Accuracy, Precision, Recall, F1-Score, and the area under the receiver operating characteristic (ROC-AUC) curve were used, along with Confusion Matrix. To address class imbalance in colorectal cancer cases, downsampling technique was implemented. Hyperparameter tuning allowed us to find the best settings for our model. We used grid search to test different combinations of settings and cross-validation to ensure the model works well on different parts of the data. The final model was selected based on its ability to generalize across unseen data while maintaining high predictive accuracy. Additionally, assess feature importance to gain deeper insights into CRC risk factors, aiming to visualize the top 13 risk factors and identify the most common key predictors using the best-performing models.

To run the provided code snippets, ensure your Python environment is set up with the necessary libraries and that the file paths to the datasets are correct. You can execute these in a Jupyter Notebook or a Python script.


Contributions to this project are welcome.

[Project Markdown File](https://github.com/JulietKuruvilla/CapstoneProject-HowardUniversity/blob/main/CapstoneCodebook_JKuruvilla.ipynb)


## Contact

For questions or further information, please contact:

# Howard University
[Email](juliet.kuruvilla1@bison.howard.edu)
